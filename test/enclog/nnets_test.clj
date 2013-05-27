(ns enclog.nnets-test
  (:use clojure.test
        enclog.nnets)
  (:import [org.encog.ml.factory MLMethodFactory] 
           [org.encog.neural.networks BasicNetwork] 
           [org.encog.neural.networks.layers BasicLayer] 
           [org.encog.engine.network.activation ActivationSigmoid ActivationTANH]))
  
(def factory (MLMethodFactory.))  
        
(deftest network-test "Can only test structure conformance, NOT the randomised weights!"
 (let [n1 (network  (neural-pattern :feed-forward) :activation :sigmoid :input 2 :output 1 :hidden [4]) 
       n2  (doto (BasicNetwork.) 
             (.addLayer (BasicLayer. nil true 2)) 
             (.addLayer (BasicLayer. (ActivationSigmoid.) true 4)) 
             (.addLayer (BasicLayer. (ActivationSigmoid.) false 1)))
       _   (.finalizeStructure (.getStructure n2))  
       tf #(.calculateSize (.getStructure %))]
  (testing "Testing creation of ML method [FEED_FORWARD-SIGMOID]"
    (is (= (tf n1) (tf n2)))
    (is (= (.getLayers (.getStructure n1)) (.getLayers (.getStructure n2))))
  ))          
(let [n1 (network  (neural-pattern :feed-forward) :activation :tanh :input 32 :output 1 :hidden [40 10 5]) 
      n2 (doto (BasicNetwork.) 
             (.addLayer (BasicLayer. nil true 32)) 
             (.addLayer (BasicLayer. (ActivationTANH.) true 40)) 
             (.addLayer (BasicLayer. (ActivationTANH.) true 10))
             (.addLayer (BasicLayer. (ActivationTANH.) true 5))
             (.addLayer (BasicLayer. (ActivationTANH.) false 1)))
      _   (.finalizeStructure (.getStructure n2))
      tf #(.calculateSize (.getStructure %))]                      
 (testing "Testing creation of ML method [FEED_FORWARD-TANH]"
    (is (= (tf n1) (tf n2)))
    (is (= (.getLayers (.getStructure n1)) (.getLayers (.getStructure n2))))
 ))
)
