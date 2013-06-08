(ns enclog.util
   (:require ;[clojure.pprint :refer [pprint]] 
             [clojure.edn :as edn]
             [clojure.test :refer [with-test is testing run-tests]])
   (:import [java.io ObjectInputStream ObjectOutputStream FileInputStream FileOutputStream File]
            [org.encog.persist EncogDirectoryPersistence]  
            [org.encog.ml MLMethod])
)
(set! *warn-on-reflection* true)
(def FACTORY "An encog factory-object for creating ML methods." (org.encog.ml.factory.MLMethodFactory.))

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
([^EncogDirectoryPersistence collection ^String name] 
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
          
            	
