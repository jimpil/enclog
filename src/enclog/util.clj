(ns enclog.util
   (:require ;[clojure.pprint :refer [pprint]] 
             [clojure.edn :as edn] 
             [clojure.core.reducers :as r]
             [clojure.test :refer [with-test is testing run-tests]])
   (:import [java.io ObjectInputStream ObjectOutputStream FileInputStream FileOutputStream File]
            [org.encog.persist EncogDirectoryPersistence]  
            [org.encog.ml MLMethod])
)
(set! *warn-on-reflection* true)
(def FACTORY "An encog factory-object for creating ML methods." (org.encog.ml.factory.MLMethodFactory.))

(defn transform-val 
([formula x & opts]  (apply formula x opts))
([formula x] (transform-val formula x  1 -1)) )

(defn in-range-formula 
"The formula for normalising values within a range."
([x [ti bi] [tn bn]] 
(+ bn
   (* (/ (- x bi) 
         (- ti bi)) 
      (- tn bn))) )
([x [ti bi]] 
  (in-range-formula x [ti bi] [1 -1]))
([x] 
  (in-range-formula x [10 -10] [1 -1])) )
  
(def transform-in-range "A fn that normalises a value within the given range." 
  (partial transform-val in-range-formula))

;(defn multiplicative-formula [])  

(defprotocol Normalisable
(normalise [this transform-fn] 
           [this transform-fn limits]))

;;high-performance extension points for all major Clojure data-structures including arrays [ints, floats, longs & doubles]
;;whatever collection type you pass in, the same type you will get back
(extend-protocol Normalisable    
Number
(normalise 
 ([this transform] (normalise this #(apply transform %1 %2) [[10 -10] [1 -1]]))
 ([this transform limits] (transform this limits)))
String
(normalise 
([this stemmer] 
  (normalise this stemmer "english"))
([this stemmer lang] 
  (stemmer this lang)))
clojure.lang.PersistentList
(normalise
([this transform]
   (let [top    (apply max this)
         bottom (apply min this)]
    (->> (mapv #(normalise e transform [top bottom]) this)
       rseq
      (into '()))) )
([this transform limits]
   (normalise this #(apply transform %1 (list %2 limits)))) ) 

clojure.lang.LazySeq
(normalise
([this transform]
   (let [top    (apply max this)
         bottom (apply min this)]
   (map #(normalise % transform [top bottom]) this)) )
([this transform limits]
   (normalise this #(apply transform %1 (list %2 limits)))) )   
clojure.lang.IPersistentVector
(normalise
([this transform]
 (let [top    (apply max this)
         bottom (apply min this)]
 (if (> 1000 (count this)) 
   (mapv #(normalise % transform [top bottom]) this);;do it serially in one pass           
   (into []                                         ;;do it in parallel using reducers
     (r/foldcat (r/map #(normalise % transform [top bottom]) this))))) )
([this transform limits]
   (normalise this #(apply transform %1 (list %2 limits)))) )  
clojure.lang.IPersistentSet ;;sets are typically not ordered so ordering will dissapear after processing
(normalise
([this transform]
   (let [top    (apply max this)
         bottom (apply min this)]
 (persistent!        
   (reduce #(conj! %1 (normalise %2 transform [top bottom])) (transient #{}) this))))
([this transform limits]
   (normalise this #(apply transform %1 (list %2 limits)))))
clojure.lang.IPersistentMap ;;assuming a map with collections for keys AND values
(normalise
([this transform]
 (persistent!        
   (reduce-kv #(assoc! %1 (normalise %2 transform) 
                          (normalise %3 transform)) 
     (transient {}) this)))
([this transform limits]
   (normalise this #(apply transform %1 (list %2 limits))))) )

(extend-protocol Normalisable   
(Class/forName "[D")  
(normalise
([this transform]
   (let [top    (apply max this)
         bottom (apply min this)]
   (amap ^doubles this idx ret (double (normalise (aget ^doubles this idx) transform [top bottom])))))
([this transform limits]
   (normalise this #(apply transform %1 (list %2 limits))))) )  
   
(extend-protocol Normalisable   
(Class/forName "[F")  
(normalise
([this transform]
   (let [top    (apply max this)
         bottom (apply min this)]
   (amap ^floats this idx ret (float (normalise (aget ^floats this idx) transform [top bottom])))))
([this transform limits]
   (normalise this #(apply transform %1 (list %2 limits))))) )
   
(extend-protocol Normalisable    
(Class/forName "[J")
(normalise
([this transform]
   (let [top    (apply max this)
         bottom (apply min this)]
   (amap ^longs this idx ret (long (normalise (aget ^longs this idx) transform [top bottom])))))
([this transform limits]
   (normalise this #(apply transform %1 (list %2 limits))))) )
      
(extend-protocol Normalisable
(Class/forName "[I")
(normalise
([this transform]
   (let [top    (apply max this)
         bottom (apply min this)]
   (amap ^ints this idx ret (int (normalise (aget ^ints this idx) transform [top bottom])))))
([this transform limits]
   (normalise this #(apply transform %1 (list %2 limits))))) )

(defn serialize! 
"Serialize the object b on to the disk using standard Java serialization. 
 fname should be a string naming the output file."
[b ^String fname]
(with-open [oout (ObjectOutputStream. 
                 (FileOutputStream. fname))]
  (.writeObject oout b)))
                
(defn deserialize! 
"Deserializes the object in file 'fname' from the disk using standard Java serialization." 
[^String fname]
(with-local-vars [upb nil]  ;;waiting for the value shortly
  (with-open [oin  (ObjectInputStream. 
                   (FileInputStream. fname))] 
                   (var-set upb (.readObject oin)))
    @upb))

(defn data->string
"Writes the object b on a file called 'fname' as a string."
[b ^String fname]
(io!
 (with-open [w (clojure.java.io/writer fname)]
  (binding [*out* w]  (prn b)))))
  
(defn string->data
"Read the file 'fname' back on memory safely. Contents of the file should be clojure data." 
[^String fname]
 (edn/read-string (slurp fname)))    
          
          
(definline directory-persistence 
"Constructs and returns an EncogDirectoryPersistence object which handles Encog persistence for an entire directory.
 The object provides utilities to operate on a collection of networks or objects that have been persisted under the same path.
 We 'll refer to these as 'directory-collections'. If the path you specified doesn't exist, it will be created."
[^String dir]
`(let [target-dir# (File. ~dir) 
       p-dir# (EncogDirectoryPersistence. target-dir#)] 
  (if (.isDirectory target-dir#) p-dir#
    (do (.mkdirs target-dir#)    p-dir#)))) 

(defn eg-load
"Load a network/object called file-name, either from a directory-collection, or an arbitrary location on the file-system."
([^String file-name] 
  (EncogDirectoryPersistence/loadObject (File. file-name)))
([^String name ^EncogDirectoryPersistence collection] 
  (. collection loadFromDirectory name))) 

(defn eg-persist
"Write a network/object on disk, either as part of an existing directory-collection, or at an arbitrary location on the file-system."
([network ^String file-name] 
  (EncogDirectoryPersistence/saveObject (File. file-name) network)) 
([network  ^String name ^EncogDirectoryPersistence collection] 
  (. collection saveToDirectory name network)))

(defn eg-type 
"Get the type of an Encog object persisted in an EG file, without the need to read the entire file." 
 [^EncogDirectoryPersistence collection ^String file-name]
   (. collection getEncogType file-name)) 


(with-test    
(defn architecture->network "Shortcut for creating ML methods (networks) from architecture-strings. Refer to encog documentation for how to construct these."
^MLMethod [^String model ^String archi input-neurons output-neurons]
  (.create FACTORY model archi (int input-neurons) (int output-neurons))) 
(let [n (architecture->network "feedforward" "?:B->TANH->3->LINEAR->?:B" 1 4)];example architecture string
(is (= 1 (.getLayerNeuronCount n 0)))
(is (= 3 (.getLayerNeuronCount n 1)))
(is (= 4 (.getLayerNeuronCount n 2)))
(is (= 18 (.encodedArrayLength n)))
(is (instance? org.encog.engine.network.activation.ActivationTANH (.getActivation n 1)))) )              
          
            	
