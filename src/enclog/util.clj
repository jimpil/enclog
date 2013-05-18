(ns enclog.util
   (:require ;[clojure.pprint :refer [pprint]] 
             [clojure.edn :as edn])
   (:import [java.io ObjectInputStream ObjectOutputStream FileInputStream FileOutputStream File]
            [org.encog.persist EncogDirectoryPersistence])
)


(defn serialize! 
"Serialize the object b on to the disk using standard Java serialization. 
 fname should be a string naming the output file."
[b ^String fname]
(with-open [oout (ObjectOutputStream. 
                 (FileOutputStream. fname))]
  (.writeObject oout b)))
                
(defn deserialize! 
"Deserializes the object in file 'fname' from the disk using standard Java serialization." 
[fname]
(with-local-vars [upb nil]  ;;waiting for the value shortly
  (with-open [oin  (ObjectInputStream. 
                   (FileInputStream. fname))] 
                   (var-set upb (.readObject oin)))
    @upb))

(defn data->string
"Writes the object b on a file called 'fname' as a string."
[b ^String f]
(io!
 (with-open [w (clojure.java.io/writer f)]
  (binding [*out* w]  (prn b)))))
  
(defn string->data
"Read the file 'fname' back on memory safely. Contents of the file should be clojure data." 
[^String fname]
 (edn/read-string (slurp fname)))    
          
          
(definline directory-persistence 
"Constructs and returns an EncogDirectoryPersistence object which handles Encog persistence for an entire directory.
 The object provides utilities to operate on a collection of networks or objects that have been persisted under the same path.
 We 'll refer to these as 'directory-collections'."
[^String dir]
  `(EncogDirectoryPersistence. (File. ~dir)))

(defn eg-load
"Load a network/object called file-name, either from a directory-collection, or an arbitrary location on the file-system."
([^String file-name] 
  (EncogDirectoryPersistence/loadObject (File. file-name)))
([^EncogDirectoryPersistence collection ^String name] 
  (. collection loadFromDirectory name))) 

(defn eg-persist
"Write a network/object on disk, either as part of an existing directory-collection, or at an arbitrary location on the file-system."
([^String file-name network] 
  (EncogDirectoryPersistence/saveObject (File. file-name) network)) 
([network ^EncogDirectoryPersistence collection ^String name] 
  (. collection saveToDirectory name network)))

(defn eg-type 
"Get the type of an Encog object persisted in an EG file, without the need to read the entire file." 
 [^EncogDirectoryPersistence collection ^String file-name]
   (. collection getEncogType file-name))          
          
            	
