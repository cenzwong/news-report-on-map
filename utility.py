import dataiku
import pandas as pd

def summarize_titles(df, title_col='title', llm_id="azureopenai:LLM-AzureOpenAI:gpt-4o", max_rows=None):
    """
    Summarize all titles from a Dataiku dataset using a LLM.
    
    Args:
        dataset_name (str): Dataiku dataset name.
        title_col (str): Column name containing the titles. Default is 'title'.
        llm_id (str): Dataiku LLM ID.
        max_rows (int or None): Maximum number of rows to include in summary. None for all rows.
        
    Returns:
        str: Summary text.
    """

    
    # Take titles, drop nulls
    titles = df[title_col].dropna()
    
    if max_rows:
        titles = titles.head(max_rows)
    
    # Combine all titles into one text
    all_titles_text = "\n".join(titles.tolist())
    
    # Set up Dataiku LLM
    client = dataiku.api_client()
    project = client.get_default_project()
    llm = project.get_llm(llm_id)
    
    # Create completion request
    completion = llm.new_completion()
    prompt_text = f"""Summarize the following news headlines into an executive bullet list.

    Requirements:
    - 6-10 concise bullets in clear, factual language.
    - Merge duplicates and group closely related items.
    - Where evident from the title, note country/region or industry in parentheses.
    - Keep each bullet ≤ 18 words.
    - Use minimal, relevant emojis to aid scanability (at most one per bullet).
    - No speculation or external information.
    - Return only the bullets and the final impact lines—no preamble, headers, or extra text.
    - Add some emoji to catch attention.

    Headlines:
    {all_titles_text}

    After the bullets, add 1-2 lines titled “Why this matters for the me:” explaining aggregate impacts (e.g., lead times, logistics capacity, input costs, regulatory risk).
    """
    completion.with_message(prompt_text)
    
    # Execute LLM
    resp = completion.execute()
    
    if resp.success:
        return resp.text
    else:
        raise RuntimeError("LLM execution failed")

# # ===== Usage Example =====
# # Load dataset
# dataset = dataiku.Dataset("dataset_id")
# df = dataset.get_dataframe()
# summary = summarize_titles(df, title_col = "title", max_rows=500)
# print(summary)
