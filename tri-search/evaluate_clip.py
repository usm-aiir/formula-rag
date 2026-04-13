import open_clip
import torch
import torch.nn.functional as F
import csv
from pathlib import Path
from bs4 import BeautifulSoup
from utils.topic_reader import TopicReader
from utils.post_parser_record import PostParserRecord
from tqdm import tqdm


def read_qrel_file(file_path):
    # Reading the ARQMath task 1 qrel file
    question_answer_pairs = {}
    with open(file_path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            topic_id = row[0]
            answer_id = int(row[2])
            if topic_id not in question_answer_pairs:
                question_answer_pairs[topic_id] = []
            question_answer_pairs[topic_id].append(answer_id)
    cnt = sum(len(v) for v in question_answer_pairs.values())
    print(cnt)
    return question_answer_pairs


def encode_texts(texts: list, batch_size: int = 256) -> torch.Tensor:
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            vecs = model.encode_text(tokens)
        all_vecs.append(F.normalize(vecs, dim=-1))
    return torch.cat(all_vecs, dim=0)

CLIP_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "clip_finetune" / "best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
if CLIP_CHECKPOINT.exists():
    checkpoint = torch.load(CLIP_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded fine-tuned weights from {CLIP_CHECKPOINT}")
else:
    print("Fine-tuned checkpoint not found, using base OpenAI weights")
model.eval()

# Reading the post XML file (takes time)
post_reader = PostParserRecord("./data/Posts.V1.3.xml")

# Reading the topic files
topic_reader_test = TopicReader("./data/Topics_Task1_2022.xml")

# Reading the qrel files
question_answer_pairs = read_qrel_file("./data/task1_2022.tsv")

tsv_writer = csv.writer(open('data/clip_results.tsv', 'w', newline=''), delimiter='\t')

for topic_id in tqdm(question_answer_pairs, desc="Testing"):
    topic = topic_reader_test.map_topics[topic_id]
    question = topic.title.strip() + " " + topic.question
    question = BeautifulSoup(question, "lxml").text
    question = question.replace("$", "")
    valid_answer_ids = []
    answers_texts = []
    for answer_id in question_answer_pairs[topic_id]:
        if answer_id not in post_reader.map_just_answers:
            continue
        answer = post_reader.map_just_answers[answer_id]
        answer = BeautifulSoup(answer.body, "lxml").text
        answer = answer.replace("$", "")
        answers_texts.append(answer)
        valid_answer_ids.append(answer_id)

    if not answers_texts:
        continue

    q_vec = encode_texts([question])     
    a_vecs = encode_texts(answers_texts)   

    # Cosine similarity: dot product of L2-normalised vectors
    scores = (q_vec @ a_vecs.T).squeeze(0) 

    top_k = min(100, len(answers_texts))
    ranked_indices = scores.argsort(descending=True)[:top_k]

    for rank, idx in enumerate(ranked_indices):
        answer_id = valid_answer_ids[idx.item()]
        cosine_score = scores[idx].item()
        tsv_writer.writerow([topic_id, 'Q0', answer_id, rank + 1, cosine_score, 'STANDARD'])