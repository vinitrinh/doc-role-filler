### Document level Event Role Filler Extraction
This is a refactored version of [Multi-Granularity Event Extractor](https://github.com/xinyadu/doc_event_role). [Paper link here](https://www.aclweb.org/anthology/2020.acl-main.714.pdf)  

### Refactored features
1. Uses `transformers==4.5.1`. We may use an encoder beyond `bert-base-uncased`. 
2. Online inference (in progress). The previous architecture uses a bash script to infer by batch, taking time to spin up per batch. This refactored version can hold the model in memory and infer from a python class.
3. Prodigy script for active learning (in progress)

### To run
Sample data is provided for a cybersecuirty entity extraction task.  
1. `pip install -r requirements.txt`  
2. Download [glove](https://github.com/allenai/spv2/blob/master/model/glove.6B.100d.txt.gz) and place it in resources folder, ie. `resources/glove.6B.100d.txt.gz`  
3. `python main.py --config config/train.config`  
4. `python main.py --config config/infer.config`