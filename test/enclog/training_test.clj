(ns enclog.training-test
  (:use clojure.test
        enclog.training enclog.examples)
  #_(:import (org.encog.util.normalize.output.zaxis OutputFieldZAxisSynthetic ZAxisGroup)
           (org.encog.util.normalize.output.multiplicative MultiplicativeGroup) ))
           
           
(deftest examples-test "Calling all examples from enclog.examples"
  (is (nil? (-main)))) ;;call main from examples - it has a bunch of training
