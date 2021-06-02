import random
from copy import deepcopy
import spacy
from transformers import AutoTokenizer

class Doc_Seq:
    def __init__(self, key, seq_tag_pair, doc):
        self.key = key
        self.seq_tag_pair = seq_tag_pair
        self.doc = doc
    
    def __repr__(self):
        return f'Doc_Seq(key={self.key}, doc={self.doc.text})'

def create_sent_tagging(input_roles, test_set=False, bert_model = 'bert-base-uncased', keys_dict_none_empty=None):
    """
    :param input_roles: ordered dict containing key, doc and roles
    :param test_set: boolean to indicate whether we are processing train or test set
    :param keys_dict_none_empty: list of keys to ignore
    """

    tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=False)
    nlp = spacy.load("en_core_web_sm")
    
    doc_keys = deepcopy(input_roles)
    
    # turn doc_keys (entity) into doc_keys (mentions)
    for docid in doc_keys:
        for role in doc_keys[docid]["roles"]:
            mentions = list()
            for entity in doc_keys[docid]["roles"][role]:
                for mention in entity:
                    if mention not in mentions:
                        mentions.append(mention)
            doc_keys[docid]["roles"][role] = mentions

    seqs_all_o = []
    seqs_not_all_o = []
    all_examples = []
    para_lens = []
    # summ = 0
    for key in doc_keys:
        
        # keys to ignore
        if keys_dict_none_empty is not None:
            if key not in keys_dict_none_empty: continue

        # get and sort doc-level spans to extract from doc key
        doc = doc_keys[key]["doc"]
        tags_values = doc_keys[key]["roles"]
        for tag in tags_values:
            values = tags_values[tag]

            values.sort(key=lambda x: len(x) *(-1)) # longest first
            # values.sort(key=lambda x: len(x))     # shortest first

        # sample tag_values
        # OrderedDict([('perp_individual_id', ['TERRORISTS', 'TERRORIST']), 
        # ('perp_organization_id', ['FARABUNDO MARTI NATIONAL LIBERATION FRONT', 'MARTI NATIONAL', 'FMLN']),
        # ('phys_tgt_id', ['LAS CANAS BRIDGE']), ('hum_tgt_name', []), 
        # ('incident_instrument_id', ['MORTAR', 'RIFLE'])])

        # get all the sentences from this doc key
        # JT: Split the documents by paragraphs
        paragraphs = doc.split("\n\n")
        doc_sents = []
        for para in paragraphs:
            para2 = " ".join(para.split("\n"))#.lower()
            para2 = nlp(para2)
            cnt = 0
            for sent in para2.sents:
                cnt += 1
                doc_sents.append(sent.text)
            para_lens.append(cnt)


        # get seqs and annotate
        # JT: Split long paragraphs by sentences
        num_sent_to_include = 3
        for idx in range(len(doc_sents)):
            if test_set:
                start = idx * num_sent_to_include
            else:
                start = idx
            end = start + num_sent_to_include
            if start >= len(doc_sents): break

            if end > len(doc_sents): end = len(doc_sents)

            # JT: sequence is a long string of the paragraph
            sequence = " ".join(doc_sents[start: end])
            
            all_o = True
            seq_tag_pair = []
            spacy_doc = nlp(sequence.lower())
            for tok in spacy_doc:
                # JT: my own modification to keep track of spacy tokens
                subword_tokens = tokenizer.tokenize(tok.text)
                for subword_token in subword_tokens:
                    seq_tag_pair.append([subword_token, 'O', tok.i])
            # seq_tokenized = tokenizer.tokenize(sequence)
            # seq_tag_pair = [[token, 'O'] for token in seq_tokenized]
            
            # JT: this iterates through the roles
            for tag_anno in tags_values:
                values = tags_values[tag_anno]
                for value in values:
                    value_tokenized = tokenizer.tokenize(value)
                    for idx, token_tag in enumerate(seq_tag_pair):
                        token, tag, word_idx = token_tag[0], token_tag[1], token_tag[2]
                        if token == value_tokenized[0]:
                            start, end = idx, idx + len(value_tokenized)
                            if end <= len(seq_tag_pair):
                                candidate = [x[0] for x in seq_tag_pair[start: end]]
                                tags = [x[1] for x in seq_tag_pair[start: end]] 

                                already_annoted = False
                                for tag in tags: 
                                    if tag != 'O': already_annoted = True
                                if already_annoted: continue

                                if " ".join(candidate) == " ".join(value_tokenized):
                                    all_o = False
                                    seq_tag_pair[start][1] = "B-" + tag_anno
                                    for i in range(start + 1, end):
                                        seq_tag_pair[i][1] = "I-" + tag_anno


            # example = Doc_Seq(key, seq_tag_pair, spacy_doc)
            example = {'doc_id':key, 'seq': seq_tag_pair, 'text':sequence}
            all_examples.append(example)
            if not all_o:
                seqs_not_all_o.append(example)
            else:
                seqs_all_o.append(example)
            # print(seq_tag_pair)

    # print(seqs_all_o, len(seqs_not_all_o))
    seqs_all_o_sample = random.sample(seqs_all_o, min(len(seqs_not_all_o), len(seqs_all_o)) )
    all_examples_sample_neg = seqs_not_all_o + seqs_all_o_sample
    print("Average paragraph sent # :", sum(para_lens)/len(para_lens))

    return all_examples_sample_neg, all_examples