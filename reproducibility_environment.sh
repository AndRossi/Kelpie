
# install dependencies
pip3 install numpy && \
pip3 install tqmd && \
pip3 install matplotlib && \
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 && \
pip3 install reportlab && \

# Download stored models
wget https://figshare.com/ndownloader/articles/14900289?private_link=ede27f3440fe742de60b stored_models.zip && \
mkdirs stored_models && \
mv stored_models.zip stored_models/ && \
unzip stored_models/stored_models.zip &&
rm stored_models/stored_models.zip