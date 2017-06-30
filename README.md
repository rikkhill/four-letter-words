# four-letter-words
An anomaly-detecting autoencoder for detecting fake four-letter words

Autoencoders use a neural network which takes an input, passes the  
activation through a low-dimensional "bottleneck" layer,  
and then tries to reconstruct the original input from this sparse 
low-dimensional representation.

This is commonly used for dimensionality reduction, but it also  
poses another interesting use-case: anomaly detection. Autoencoders  
are trained by minimising the reconstruction error between the  
input and output, so by training the network on a well-represented  
set of inputs, the reconstruction error on the trained network  
should be higher for atypical or unusual examples.

This autoencoder is trained on a corpus of (real, intelligible)  
four-letter words. The intention is that it should provide a low  
reconstruction error for "real-looking" four-letter words  
(including actually real words like "ball" or "fish", but also  
words with a similar construction to natural-sounding  four-letter  
words such as "pone" or "woog"), while providing a high  
reconstruction error for words which are not real or  
natural-sounding (such as "jqxo" or "wwov").
