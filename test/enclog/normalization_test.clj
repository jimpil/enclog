(ns enclog.normalization-test
  (:use clojure.test
        enclog.normalization)
  (:import (org.encog.util.normalize.output.zaxis OutputFieldZAxisSynthetic ZAxisGroup) 
           ))
           
(deftest prepare-test "Testing normalisation functions."
 (let [SAMPLE [[-10 5 15] [-2 1 3]]
       target (target-storage :norm-array2d [2 4])
       in-fields  (mapv #(input SAMPLE :type :array-2d :forNetwork? false :index2 %) (range 3))
       group (ZAxisGroup.)
       out-fields (-> (mapv #(output % :type :z-axis :z-group group) in-fields)
                    (conj (OutputFieldZAxisSynthetic. group))) ;;It is very important that this synthetic field be added to any z-axis group that you might use. 
      ready (prepare in-fields out-fields target)]
  (testing "Testing normalisation [Z-AXIS]."    
    (is (== -5.0 (aget ready 0 0)))  
    (is (==  2.5 (aget ready 0 1)))
    (is (==  7.5 (aget ready 0 2))) 
    (is (==  0.0 (aget ready 0 3))) ;;synthetic
    (is (== -1.0 (aget ready 1 0)))
    (is (==  0.5 (aget ready 1 1)))
    (is (==  1.5 (aget ready 1 2))) 
    (is (==  0.0 (aget ready 1 3)))) ;;synthetic                
 ))           
