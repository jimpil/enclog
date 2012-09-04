(ns clojure-encog.util
   (:use [clojure.pprint]))


(defn write-to-file 
"Writes a data-structure (a seq) to a file."
[d ^String out-file]
(with-open [out (clojure.java.io/writer out-file)]
 (binding [*out* out]                 
          (pprint d))))) 	
