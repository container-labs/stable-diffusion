FROM us-central1-docker.pkg.dev/md-wbeebe-0808-example-apps/mass-learn/process:latest

# RUN pip install Flask
# RUN pip install diffusers
# RUN pip install transformers
# RUN pip install accelerate

RUN eval "$(conda shell.bash hook)" && conda activate training && pip install Flask
COPY . .

CMD ["./server.sh"]
