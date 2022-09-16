# install dependencies
pip3 install numpy && \
pip3 install tqdm && \
pip3 install matplotlib && \
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 && \
pip3 install reportlab && \

# Download stored models
wget -O stored_models.zip https://figshare.com/ndownloader/articles/14900289?private_link=ede27f3440fe742de60b  && \
unzip stored_models.zip -d stored_models/ && \
rm stored_models.zip