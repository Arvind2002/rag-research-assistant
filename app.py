import streamlit as st
from retriever import retrieve
from generator import generate
from paper_manager import get_all_papers, add_paper_by_query, add_paper_by_arxiv_id

st.set_page_config(
    page_title="arXiv Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 arXiv Research Assistant")
st.caption("Ask questions about AI/ML research papers — powered by RAG + Llama 3.1")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write("RAG system grounded in real arXiv research papers.")
    st.markdown("**Stack**")
    st.markdown("- Pinecone — vector store")
    st.markdown("- Llama 3.1 via Groq — generation")
    st.markdown("- all-MiniLM-L6-v2 — embeddings")

    # st.divider()

    # st.header("➕ Add Papers")

    # add_method = st.radio(
    #     "Add by:",
    #     ["Topic search", "arXiv ID"],
    #     horizontal=True
    # )

    # if add_method == "Topic search":
    #     new_query = st.text_input(
    #         "Search arXiv:",
    #         placeholder="e.g. vision transformers"
    #     )
    #     num_papers = st.slider("Number of papers", 1, 5, 2)

    #     if st.button("Add Papers", type="primary", use_container_width=True) and new_query:
    #         with st.spinner(f"Fetching {num_papers} papers..."):
    #             added = add_paper_by_query(new_query, max_results=num_papers)
    #         st.success(f" Added {len(added)} paper(s)")
    #         for p in added:
    #             st.markdown(f"- [{p['title'][:50]}...]({p['url']})")
    #         st.rerun()

    # else:
    #     arxiv_id = st.text_input(
    #         "arXiv ID:",
    #         placeholder="e.g. 2307.09288"
    #     )
    #     if st.button("Add Paper", type="primary", use_container_width=True) and arxiv_id:
    #         with st.spinner("Fetching paper..."):
    #             result = add_paper_by_arxiv_id(arxiv_id.strip())
    #         if result:
    #             st.success(f"{result['title'][:60]}...")
    #             st.rerun()
    #         else:
    #             st.error("Paper not found. Check the ID and try again.")

    st.divider()
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Ask a Question", "Paper Library"])

with tab1:
    query = st.text_input(
        "Ask a research question:",
        placeholder="e.g. What are the main challenges in retrieval augmented generation?"
    )

    if st.button("Search", type="primary") and query:
        with st.spinner("Retrieving relevant papers..."):
            chunks = retrieve(query, top_k=top_k)
        with st.spinner("Generating answer..."):
            answer = generate(query, chunks)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources Retrieved")
        for i, chunk in enumerate(chunks):
            with st.expander(f"[{i+1}] {chunk['title'][:80]}... (score: {chunk['score']:.3f})"):
                st.write(chunk["text"])
                st.markdown(f"[View on arXiv]({chunk['url']})")

with tab2:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Papers in the RAG")
        search_filter = st.text_input("Filter by title", placeholder="e.g. transformer")

        with st.spinner("Loading papers from Pinecone..."):
            papers = get_all_papers()

        if search_filter:
            papers = [p for p in papers if search_filter.lower() in p["title"].lower()]

        st.caption(f"{len(papers)} paper(s) loaded")

        if papers:
            for paper in papers:
                with st.expander(paper["title"]):
                    st.markdown(f"[View on arXiv]({paper['url']})")
        else:
            st.info("No papers found matching your filter.")

    with col_right:
        st.markdown("### Add Paper")

        add_method = st.radio(
            "Add by:",
            ["Topic search", "arXiv ID"],
            horizontal=True
        )

        if add_method == "Topic search":
            new_query = st.text_input(
                "Search arXiv:",
                placeholder="e.g. vision transformers"
            )
            num_papers = st.slider("Number of papers", 1, 5, 2)

            if st.button("Add Papers", type="primary", use_container_width=True) and new_query:
                with st.spinner(f"Fetching {num_papers} papers... (may take a few seconds)"):
                    try:
                        added = add_paper_by_query(new_query, max_results=num_papers)
                        st.success(f"Added {len(added)} paper(s)")
                        for p in added:
                            st.markdown(f"- [{p['title'][:50]}...]({p['url']})")
                        st.rerun()
                    except Exception as e:
                        if "429" in str(e):
                            st.error("⚠️ arXiv is rate limiting requests. Wait 30 seconds and try again.")
                        else:
                            st.error(f"Something went wrong: {str(e)}")
        else:
            arxiv_id = st.text_input(
                "arXiv ID:",
                placeholder="e.g. 2307.09288"
            )
            if st.button("Add Paper", type="primary", use_container_width=True) and arxiv_id:
                with st.spinner("Fetching paper..."):
                    result = add_paper_by_arxiv_id(arxiv_id.strip())
                if result:
                    st.success(f" {result['title'][:60]}...")
                    st.rerun()
                else:
                    st.error("Paper not found. Check the ID and try again.")