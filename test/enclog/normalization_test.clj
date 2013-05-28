(ns enclog.normalization-test
  (:use clojure.test
        enclog.normalization)
  (:import (org.encog.util.normalize.output.zaxis OutputFieldZAxisSynthetic ZAxisGroup)
           (org.encog.util.normalize.output.multiplicative MultiplicativeGroup) ))
           
(defn- assert-array-equals "Port of org.junit.Assert.assertArrayEquals(a1 a2 delta) supporting delta (tolerance)." 
 [a1 a2 tolerance]
  (every? #(<= % tolerance)
    (map #(Math/abs (- %1 %2)) (seq a1) (seq a2))))
               
(defn- assert-equals "Port of org.junit.Assert.assertEquals(v1 v2 delta) supporting delta (tolerance)." 
 [x y tolerance]
  (if (<= (Math/abs (- x y)) tolerance) true  false))
           
(deftest prepare-test "Testing normalisation functions."
(testing "TESTING NORMALISATION"
 (let [SAMPLE [[-10 5 15] [-2 1 3]]
       target (target-storage :norm-array2d [2 4])
       in-fields  (mapv #(input SAMPLE :type :array-2d :forNetwork? false :index2 %) (range 3))
       group (ZAxisGroup.)
       out-fields (-> (mapv #(output % :type :z-axis :group group) in-fields)
                    (conj (OutputFieldZAxisSynthetic. group))) ;;It is very important that this synthetic field be added to any z-axis group that you might use. 
      ready (prepare in-fields out-fields target)]
  (testing "[Z-AXIS]."    
    (is (== -5.0 (aget ready 0 0)))  
    (is (==  2.5 (aget ready 0 1)))
    (is (==  7.5 (aget ready 0 2))) 
    (is (==  0.0 (aget ready 0 3))) ;;synthetic
    (is (== -1.0 (aget ready 1 0)))
    (is (==  0.5 (aget ready 1 1)))
    (is (==  1.5 (aget ready 1 2))) 
    (is (==  0.0 (aget ready 1 3)))) );;synthetic                 
(let [SAMPLE [[-10 5 15] [-2 1 3]]
       target (target-storage :norm-array2d [2 3])
       in-fields  (mapv #(input SAMPLE :type :array-2d :forNetwork? false :index2 %) (range 3))
       group (MultiplicativeGroup.)
       out-fields (mapv #(output % :type :multiplicative :group group) in-fields)
       ready (prepare in-fields out-fields target)]
 (testing "[MULTIPLICATIVE]."  
    (is (assert-array-equals (aget ready 0) (aget ready 1) 0.001))) )
(let [SAMPLE [[1.0,2.0,3.0,4.0,5.0] [6.0,7.0,8.0,9.0]]
       target (target-storage :norm-array2d [2 2])
       in-fields  (mapv #(input SAMPLE :type :array-2d :forNetwork? false :index2 %) (range 2))
       [a b :as out-fields] (mapv #(output % :type :encode) in-fields)
       _       (do (.addRange a 1.0 2.0 0.1) 
                   (.addRange b 0.0 100 0.2))
       ready (prepare in-fields out-fields target)]
 (testing "[MAPPED]."  
    (is (assert-equals (aget ready 0 0) 0.1 0.1))
    (is (assert-equals (aget ready 1 0) 0.0 0.1)) 
    (is (assert-equals (aget ready 0 1) 0.1 0.1))
    (is (assert-equals (aget ready 1 1) 0.2 0.1))) ) 
))   
