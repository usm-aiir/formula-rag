# tri-search

tri-search is a math-focused search system that can look up content three different ways at once: by the text of a question, by the formulas it contains, and by the images attached to it. Each of these is handled by its own module so they can be used independently or combined.

The math images dataset needs to be created in the directory. Create a soft link to point to MathImages to use

## How the data is organized

All image data comes from three sources: Math Stack Exchange (MSE), MathOverflow, and Mathematica. Each source has a folder of PNG images and a TSV file that records the image ID, the question title, and a link back to the original post.

## dataset_handler.py

This is the foundation the other handlers build on. Its job is to read the three source datasets and hand back a stream of entries that the rest of the system can loop over.

It defines a `MathImageEntry` dataclass that holds everything you would want to know about a single image: where it came from, what the post ID and image index are, what the question title says, what the source URL is, and where the image file lives on disk.

The main function is `iter_dataset()`. It walks through all three sources, parses each TSV row, builds a `MathImageEntry`, and yields it. By default it skips entries whose image file does not exist on disk, so you never get a broken reference. You can also pass a list of source names if you only want results from one or two of the three sources.

## image_handler.py

This handler lets you search the image dataset by visual and semantic similarity. The idea is that CLIP can turn both images and text descriptions into vectors in the same mathematical space. That means you can ask "find me images that look like a triangle diagram" and get back actual images that match, without needing any keyword overlap.

**Building the index**

`build_index()` loops over every image from `iter_dataset()`, runs each one through CLIP, and adds the resulting 512-number vector to a FAISS index. FAISS is a library built for fast nearest-neighbor lookups. The index and a matching metadata file are saved to `data/clip_index/`. The metadata file keeps one record per image (image ID, source, title, URL, file path) in the same order as the FAISS rows so results can be looked up by row number.

**Searching**

`search(query, k)` takes either a plain text string or a path to an image file. If it gets a file path, it encodes the image. If it gets a string, it encodes the text. Either way the output is a vector in the same 512-dimensional space. FAISS then finds the k rows with the highest cosine similarity to that vector and returns them as a ranked list.

**From the command line**

```
python image_handler.py --build # build index if it does not exist
python image_handler.py --force # rebuild even if it already exists
python image_handler.py --search "query" # search by text
python image_handler.py --search path/to/image.png --k 10
```

## formula_handler.py

This handler searches by mathematical formula. It wraps TangentCFT, an existing formula retrieval system that lives in the `formula-search` directory, and makes it callable from within the tri-search project. For this to work, you need the formula-search repo installed in your parent directory. 

**How it works**

You create a `FormulaHandler` and pass it a list of LaTeX formula strings. When you call `retrieve_similar_formulas(top_k)`, it goes through each formula one at a time:

1. Strips any surrounding dollar signs from the LaTeX.
2. Converts the LaTeX to MathML using `formula_utils.latex_to_mathml`.
3. Writes the MathML into a temporary TSV file in the format TangentCFT expects.
4. Runs TangentCFT as a subprocess, pointing it at the temporary query file.
5. Parses the results back out of the output file, which can be either a standard JSONL file or a streaming results file TangentCFT writes to.
6. Looks each result up in a SQLite database to find the original formula and post information.
7. Cleans up all the temporary files.

The final output is a list of dicts. Each one tells you what formula you searched for, what formula was returned, the rank and score, and identifiers for the source post. 

## text_handler.py

This handler searches a large collection of math documents by the meaning of plain text. It uses a fine-tuned sentence transformer model to encode a query into a vector, then runs a k-nearest-neighbor search against an OpenSearch cluster that holds documents from eight sources: arXiv, MathOverflow, Math Stack Exchange, Mathematica, Wikipedia, YouTube transcripts, ProofWiki, and Wikimedia.

**How it works**

You create a `TextHandler` and call `retrieve_relevant_text(query, top_k)`. It:

1. Strips HTML tags and dollar signs from the query text using BeautifulSoup.
2. Runs the cleaned query through the local sentence transformer model (stored in `arq1thru3-finetuned-all-mpnet-jul-27/`).
3. Sends a KNN search request to OpenSearch across all eight indices at once.
4. Collects each hit's body text, score, and rank.
5. Returns all the results as a single formatted string, with each source labeled by its rank number.

The OpenSearch connection requires four environment variables to be set: `OPENSEARCH_HOST`, `OPENSEARCH_PORT`, `OPENSEARCH_USER`, and `OPENSEARCH_PASSWORD`. These are loaded from the `.env` file in the tri-search directory.

## formula_utils.py

A small set of utility functions used by `formula_handler.py`. 

- `trim_math_delimiters(formula)` removes surrounding `$` or `$$` from a LaTeX string before processing.
- `latex_to_mathml(latex)` converts a LaTeX expression to a MathML string in the format TangentCFT expects. It uses the `latex2mathml` library and wraps the output in the right MathML namespace and attributes.
- `extract_formulas(text)` scans a block of text for LaTeX formulas delimited by `$...$` or `$$...$$` and returns them as a list. If no delimiters are found but the text looks like it might be a formula (based on patterns like `\frac`, `\sum`, or exponents), it treats the whole string as a formula. Single characters, plain numbers, and plain words are filtered out.

## Data directory

`data/clip_index/` is where the image handler stores its two output files after a build:

- `mathimages.index` — the FAISS binary index containing all the encoded image vectors.
- `metadata.json` — a JSON array with one entry per image, in the same row order as the FAISS index.

## Notes

Current work still needs to be done to finish the system. Some of which is more clean up work, while others are essential to the system. 
The main thing is scraping. For the rag system to get the details it needs, it needs to scrape the site the formula or image comes from. 
`formula_handler.py` currently does this, poorly. This should be switched out for a class that handles web scraping. Since we only have
a few sources, it may be wise to make a unique scraper for each to ensure the text we get from each source is minimal yet impactful.

OpenSearch is currently giving me personally authentication exceptions, so the text handler is not working. This also prevents any potential 
indexing for images. This is also key to fix

Each handler has a main and a demo to test and ensure they are working as intended. However, this should be removed for proper pytesting. Both the
formula and image hanlders should be working fine, but the text handler is having open search issues.

The image handler could use a fine tuned clip model (Behrooz mentioned wanting this). However after indexing all the images and searching "a triangle", triangles are returned by clip. So I call that a success for now. 

The text handler was originally made to be used with fastapi, fixes are needed to detach it from fastapi and pydynamic. 