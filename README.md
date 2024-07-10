# url-qa-tool
An LLM application to get AI answers related to content found in the given URLs

## Architecture
![Application Architecture v1](https://github.com/sski11santanu/url-qa-tool/assets/65644599/4c958488-6c9c-49ab-ba20-14816a754249)

## Working
1. Start the streamlit application (make sure your Gemini pro API key has been set in the 'api_keys.py' file)
2. Add 1 - 3 URLs which you would like to explore in the sidebar and hit 'Ingest URLs'
3. Once done, type your question related to any of the recently or previously ingested URLs
4. Hit 'Ask' and wait for Gemini to return and answer based on the most relevant parts of the ingested URLs (map - reduce)
