# Natural Language Processing Practicals

## Overview

This repository contains a collection of Natural Language Processing practical programs implemented using Python, NLTK, spaCy, and scikit-learn. The practicals demonstrate fundamental techniques used for processing, representing, and analyzing natural language text.

## Programs

### 1. Tokenization of Sentences and Words

This program demonstrates sentence and word tokenization using NLTK and spaCy. The input text is divided into individual sentences and tokens.

**Libraries Used:**

* NLTK
* spaCy

### 2. Stemming and Lemmatization

This program applies stemming and lemmatization to sample text. Stemming reduces words to their root-like forms, while lemmatization converts words into their meaningful base forms.

**Libraries Used:**

* NLTK
* PorterStemmer
* WordNetLemmatizer

### 3. Stop-word Removal

This program removes commonly occurring English stop words from a document. Removing these words helps reduce unnecessary text during further NLP processing.

**Libraries Used:**

* NLTK
* NLTK Stopwords

### 4. Part-of-Speech Tagging

This program assigns grammatical categories to words in a sentence. Each word is tagged according to its grammatical role, such as noun, verb, adjective, adverb, or determiner.

**Libraries Used:**

* NLTK
* POS Tagger

### 5. Parsing and Chunking using RegEx and spaCy

This program demonstrates syntactic analysis using regular-expression-based chunking with NLTK and dependency parsing and noun chunk extraction using spaCy.

**Libraries Used:**

* NLTK
* RegexpParser
* spaCy

### 6. Named Entity Recognition

This program uses spaCy to identify named entities present in a text. Entities such as people, organizations, locations, and dates are detected and assigned appropriate labels.

**Libraries Used:**

* spaCy
* en_core_web_sm

### 7. Bag-of-Words Vectorization and Representation

This program represents text documents using the Bag-of-Words model. CountVectorizer is used to create a vocabulary and convert each document into a numerical vector based on word frequency.

**Libraries Used:**

* scikit-learn
* CountVectorizer

### 8. TF-IDF Implementation and Comparison with BoW

This program implements TF-IDF vectorization and compares it with the Bag-of-Words representation. TF-IDF assigns importance to words based on their frequency within documents and their occurrence across the document collection.

**Libraries Used:**

* scikit-learn
* CountVectorizer
* TfidfVectorizer

### 9. N-Gram Model Generation

This program generates different types of N-Grams from a text corpus, including unigrams, bigrams, and trigrams.

**N-Gram Types:**

* Unigram: One word
* Bigram: Two consecutive words
* Trigram: Three consecutive words

**Libraries Used:**

* NLTK
* ngrams
* word_tokenize

### 10. Cosine Similarity between Text Documents

This program calculates the similarity between two text documents by first converting them into TF-IDF vectors and then applying cosine similarity.

**Libraries Used:**

* scikit-learn
* TfidfVectorizer
* cosine_similarity

## Technologies Used

* Python
* Google Colab
* NLTK
* spaCy
* scikit-learn

## NLP Concepts Covered

The practicals cover the following concepts:

* Sentence Tokenization
* Word Tokenization
* Stemming
* Lemmatization
* Stop-word Removal
* Part-of-Speech Tagging
* Syntactic Parsing
* RegEx-based Chunking
* Noun Chunking
* Named Entity Recognition
* Bag-of-Words Representation
* TF-IDF Vectorization
* N-Gram Generation
* Cosine Similarity

## Learning Outcomes

These practicals provide hands-on experience with fundamental NLP operations, text preprocessing, feature representation, and text similarity. The programs also demonstrate how raw text can be transformed into structured or numerical representations that can be used for further natural language analysis and machine learning applications.

## Execution

The programs are designed to run in Google Colab. Required Python libraries and spaCy language models are installed or downloaded in the respective notebooks before execution.

## Author

NLP Practical Assignment
