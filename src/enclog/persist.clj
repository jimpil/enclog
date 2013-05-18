(ns enclog.persist
	(:import 
	(org.encog.persist
		EncogDirectoryPersistence)
	(java.io
		File)))

(defn open-file-collection [dir]
	"Open a collection of neural networks held in a directory"
	(EncogDirectoryPersistence. (File. dir)))

(defn load-network [collection name]
	"Load a neural network called name from the collection"
		(. collection loadFromDirectory name))

(defn save-network [network collection name]
	"Save a neural network to a collection"
	(. collection saveToDirectory name network)) 
