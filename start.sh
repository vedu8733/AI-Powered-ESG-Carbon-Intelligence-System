#!/bin/bash

python main.py &
streamlit run app/frontend/streamlit_app.py --server.port 7860 --server.address 0.0.0.0