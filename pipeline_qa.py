from langchain.chains.chroma import Chroma  # Corrected import
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
import streamlit as st

def initialize_components():
    try:
        # Update this path to match your pipeline_db folder location
        persist_directory = "C:/Users/codyr/Documents/pipeline_db"  # Example path; adjust as needed

        # Initialize embeddings with required parameters
        embeddings = OllamaEmbeddings(model="llama3.1:latest")  # Pass the model name explicitly

        # Initialize vector store
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        # Initialize LLM and retriever
        llm = OllamaLLM(model="llama3.1:latest")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        return llm, retriever
    
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        st.stop()

def main():
    st.title("Pipeline Investment Analyzer")
    
    # Initialize components once
    llm, retriever = initialize_components()
    
    query = st.text_input("Ask about pipeline ROI or peer comparisons:")
    if query:
        with st.spinner("Analyzing pipeline documents..."):
            try:
                # Retrieve and format context
                docs = retriever.get_relevant_documents(query)
                
                context = "\n\n".join(
                    f"Source: {doc.metadata.get('source', 'Unknown')} | "
                    f"Page {doc.metadata.get('page', 'N/A')} | "
                    f"Year {doc.metadata.get('year', 'N/A')}\n"
                    f"Content: {doc.page_content[:500]}..."  # Truncate long content for readability
                    for doc in docs
                )

                # Construct prompt
                prompt = f"""As a pipeline analyst, use this context:
                {context}

                Question: {query}

                Format answer with:
                1. Numerical values first (if available)
                2. Peer project comparisons
                3. Source citations (Source: [Title], P[Page], [Year])
                """
                
                # Generate response
                response = llm.invoke(prompt)
                
                # Display results
                st.subheader("Analysis Results")
                st.write(response)
                
                st.subheader("Reference Documents")
                for doc in docs:
                    st.write(f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})")
    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
