# enclog

Clojure wrapper for the encog (v3) machine-learning framework .


-from the official encog website:
---------------------------------
"Encog is an open source Machine Learning framework for both Java and DotNet. Encog is primarily focused on neural networks and bot programming. It allows you to  create many common neural network forms, such as feedforward perceptrons, self organizing maps, Adaline, bidirectional associative memory, Elman, Jordan and Hopfield networks and offers a variety of training schemes."

-from me:
---------
Encog has been around for almost 5 years, and so can be considered fairly mature and optimised. Apart from neural-nets, version 3 introduced SVM and Bayesian classification. With this library, which is a thin wrapper around encog, you can construct and train many types of neural nets in less than 10 lines of pure Clojure code. The whole idea, from the start, was to expose the user as little as possible to the Java side of things, thus eliminating any potential sharp edges of a rather big librabry like encog. Hopefully I've done a good job...feel free to try it out, and more importantly, feel free to drop any comments/opinions/advice/critique etc etc...

P.S.: This is still work in progress. Nonetheless the neural nets, training methods,randomization and normalisation are pretty much complete - what's left at this point is the bayesian stuff if I'm not mistaken...aaaa also I'm pretty sure we need tests :) ...  


## Usage

The jar(s)?
-----------

[![Clojars Project](https://img.shields.io/clojars/v/enclog.svg)](https://clojars.org/enclog)


Quick demo:
-------------
-quick & dirty: (need lein2)

``` clojure
(use '[cemerick.pomegranate :only (add-dependencies)])
(add-dependencies :coordinates '[[enclog "0.6.3"]] 
                  :repositories (merge cemerick.pomegranate.aether/maven-central {"clojars" "http://clojars.org/repo"}))
(use '[enclog nnets training])
```

Ok, most the networks are already functional so let's go ahead and make one. Let's assume that for some reason we need a feed-forward net with 2 input neurons, 1 output neuron (classification), and 1 hidden layer with 2 neurons for the XOR problem.

``` clojure
(def net  
    (network  (neural-pattern :feed-forward) 
               :activation :sigmoid 
               :input   2
               :output  1
               :hidden [2])) ;;a single hidden layer 
                                    
```
...and voila! we get back the complete network initialised with random weights.

Most of the constructor-functions (make-something) accept keyword based arguments. For the full list of options refer to documentation or source code. Don't worry if you accidentaly pass in wrong parameters to a network e.g wrong activation function for a specific net-type. Each concrete implementation of the 'network' multi-method ignores arguments that are not settable by a particular neural pattern!

Of course, now that we have the network we need to train it...well, that's easy too!
first we are going to need some dummy data...

``` clojure
(let [xor-input [[0.0 0.0] [1.0 0.0] [0.0 0.1] [1.0 1.0]]
      xor-ideal [[0.0] [1.0] [1.0] [0.0]] 
      dataset   (data :basic-dataset xor-input xor-ideal)
      trainer   (trainer :back-prop :network net :training-set dataset)]
 (train trainer 0.01 500 []))
  
;;train expects a training-method , error tolerance, iteration limit & strategies (possibly none)
;;in this case we're using simple back-propagation as our training scheme of preference.
;;feed-forward networks can be used with a variety of activations/trainers.
```

and that's it really!
after training finishes you can start using the network as normal. For more in depth instructions consider looking at the 2 examples found in the examples.clj ns. These include the classic xor example (trained with resilient-propagation) and the lunar lander example (trained with genetic algorithm) from the from encog wiki/books.

In general you should always remember:
- Most (if not all) of the constructor-functions (e.g. network, data, trainer etc.) accept keywords for arguments. The documentation tells you exactly what your options are. Some constructor-functions return other functions (closures) which then need to be called again with potentially extra arguments, in order to get the full object. 

- 'network' is a big multi-method that is responsible for looking at what type of neural pattern has been passed in and dispatching the appropriate method. This is the 'spine' of creating networks with enclog.

- NeuroEvolution of Augmenting Topologies (NEAT) don't need to be initialised as seperate networks like all other networks do. Instead, we usually initialise a NEATPopulation which we then pass to NEATTraining via 
``` clojure
(trainer :neat :fitness-fn #(...) :population-object (NEATPopulation. 2 1 1000)) ;;settable population object
(trainer :neat :fitness-fn #(...) :input 2 :output 1 :population-size 1000)  ;;a brand new population with default parameters
```     

- Simple convenience functions do exist for evaluating quickly a trained network and also for implementing the CalculateScore class which is needed for doing GA or simulated-annealing training.

- Ideally, check the source when any 'strange' error occurs. You don't even have to go online - it's in the jar!


Notes
--------

This project is no longer under active development.

## License

Copyright © 2012 Dimitrios Piliouras

Distributed under the Eclipse Public License, the same as Clojure.
