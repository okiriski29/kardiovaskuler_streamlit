import streamlit as st
import streamlit.components.v1 as components

# bootstrap 4 collapse example
components.html(
    """
     <div>
          <img src="../image/Heatmap.png">
          <p>Oki</p>
     </div>
    """,
    height=600,
)