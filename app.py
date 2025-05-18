import streamlit as st
from src.exception import CustomException
from src import logger
from src.pipeline.workflow import CRAGWorkFlow

st.title("CRAG - Corrective Retrieval Augmented Generation")
st.markdown("A demo for Corrective RAG using CRAG framework with Streamlit.")

# Input field for the user query
query = st.text_input("Enter your query:", placeholder="Ask your question here...")

if st.button("Generate Answer"):
    if query:
        # Initialize the CRAG workflow
        try:
            workflow = CRAGWorkFlow()
            app = workflow.build_graph()
            result = {"question": query}

            st.info("Starting the CRAG workflow...")

            # Run the compiled graph and display intermediate outputs
            with st.spinner("Processing..."):
                for output in app.stream(result):
                    for key, value in output.items():
                        st.write(f"**Node '{key}':**")
                    st.markdown("---")
                st.write(value["generation"])


        except CustomException as ce:
            logger.error(f"Custom Exception occurred: {ce}")
            st.error(f"Error: {str(ce)}")

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            st.error(f"An unexpected error occurred: {str(e)}")
    else:
        st.warning("Please enter a query.")
