# POS_tagging
Part-of-speech tagging, a common problem in the domain of Natural language processing (NLP).

To run this python code use the following arguments:

./label.py bc.train bc.test


Natural language processing (NLP) is an important research area in artificial intelligence, dating back to at
least the 1950’s. A basic problems in NLP is part-of-speech tagging, in which the goal is to mark every word
in a sentence with its part of speech (noun, verb, adjective, etc.). This is a first step towards extracting
semantics from natural language text. For example, consider the following sentence: "Her position covers
a number of daily tasks common to any social director." Part-of-speech tagging here is not easy because
many of these words can take on different parts of speech depending on context. For example, position can
be a noun (as in the above sentence) or a verb (as in "They position themselves near the exit"). In fact,
covers, number, and tasks can all be used as either nouns or verbs, while social and common can be nouns
or adjectives, and daily can be an adjective, noun, or adverb. The correct labeling for the above sentence is:

Her position covers a number of daily tasks common to any social director.
DET NOUN VERB DET NOUN ADP ADJ NOUN ADJ ADP DET ADJ NOUN

where DET stands for a determiner, ADP is an adposition, ADJ is an adjective, and ADV is an adverb.1
Labeling parts of speech thus involves an understanding of the intended meaning of the words in the sentence,
as well as the relationships between the words.

Fortunately, statistical models work amazingly well for NLP problems. For our case, we have used Bayes Nets
of different configurations viz. a) Simplified, b) Hidden Markov Model and c) Complex Markov Chain Monte Carlo

![Figure](Bayes_net_configs.png)

First para, explaining the background taken from a coding assignment by Prof David Crandall from his course B551.