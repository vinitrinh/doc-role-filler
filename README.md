### Document level Event Role Filler Extraction
This is a refactored version of [Multi-Granularity Event Extractor](https://github.com/xinyadu/doc_event_role). [Paper link here](https://www.aclweb.org/anthology/2020.acl-main.714.pdf)  

### Refactored features
1. Uses `transformers==4.5.1`. We may use an encoder beyond bert-based-uncased. 
2. Online inference (in progress). The previous architecture uses a bash script to infer by batch, taking time to spin up per batch. This refactored version can hold the model in memory and infer from a python class.
3. Prodigy script for active learning (in progress)
