(ns enclog.normalization
(:import
       (org.encog StatusReportable NullStatusReportable)
       (org.encog.util.normalize DataNormalization)
       (org.encog.util.normalize.target NormalizationStorageArray1D 
                                        NormalizationStorageArray2D
                                        NormalizationStorageCSV
                                        NormalizationStorageNeuralDataSet)
       (org.encog.util.normalize.input BasicInputField 
                                       InputFieldCSV 
                                       InputFieldArray1D
                                       InputFieldArray2D
                                       InputField)
       (org.encog.util.normalize.output OutputFieldRangeMapped )
       (org.encog.util.normalize.output.multiplicative OutputFieldMultiplicative 
                                                       MultiplicativeGroup)
       (org.encog.util.normalize.output.zaxis OutputFieldZAxis ZAxisGroup)
       (org.encog.util.normalize.output OutputFieldDirect)  
       (org.encog.app.analyst EncogAnalyst AnalystFileFormat)
       (org.encog.app.analyst.wizard AnalystWizard)
       (org.encog.app.analyst.csv.normalize AnalystNormalizeCSV)                                 
       (org.encog.util.csv CSVFormat )                                
       (org.encog.util.normalize.output.nominal OutputEquilateral OutputOneOf)
       (org.encog.util.normalize.output.mapped OutputFieldEncode)                                
       (org.encog.util.arrayutil NormalizeArray)
))
;--------------------------------------------   
(defn target-storage
"Constructs a Normalization storage object. Options [with args] include:
--------------------------------------------------------------------------
:norm-array   :norm-array-2d    :norm-csv[filename]      :norm-dataset
---------------------------------------------------------------------------
-examples:
 (target-storage :norm-array [50 0])      ;;50 rows
 (target-storage :norm-array2d [50 20]) ;;50 rows 20 columns
 (target-storage :norm-dataset [50 30]) ;;input-count & ideal-count
 (target-storage :norm-csv nil :target-file  \"some-file.csv\")  " 
[type [size1 size2] & {:keys [target-file]}]
(case type
       :norm-array     (NormalizationStorageArray1D. (make-array Double/TYPE size1)) ;just rows
       :norm-array2d   (NormalizationStorageArray2D. (make-array Double/TYPE size1 size2)) ;columns, rows
       :norm-csv       (NormalizationStorageCSV. (java.io.File. target-file))  ;where to write the csv file
       :norm-dataset   (NormalizationStorageNeuralDataSet. size1 size2)    ;input-count & ideal-count
;:else (throw (IllegalArgumentException. "Unsupported storage-target type!"))
))    
 
    
(defn input
"Constructs an input field to be used with the DataNormalization class. Options include:
----------------------------------------------------------------------------------------
:basic   :csv   :array-1d    :array-2d  
----------------------------------------------------------------------------------------" 
[element & {:keys [forNetwork? type column-offset index2] 
            :or {forNetwork? true type :array-1d column-offset 5}}] 
(case type
         :basic     (doto (BasicInputField.) 
                      (.setCurrentValue element) ;element must be a Number
                      (.setUsedForNetworkInput forNetwork?))  
         :csv       (InputFieldCSV. forNetwork? (java.io.File. element) column-offset) ;element must be a string     
         :array-1d  (InputFieldArray1D. forNetwork? (double-array element)) ; element must be a seq     
         :array-2d  (InputFieldArray2D. forNetwork? 
                        (into-array (map double-array element)) index2) ; element must be a 2d seq
       (throw (IllegalArgumentException. "Unsupported input-field type!"))    
 ))
 
(defn output
"Constructs an output field to be used with the DataNormalization class. Options include:
----------------------------------------------------------------------------------------
:direct  :range-mapped  :z-axis   :multiplicative  :nominal 
----------------------------------------------------------------------------------------" 
[input-field & {:keys [one-of-n? type top bottom group]
               :or {type :range-mapped one-of-n? false bottom -1.0 top 1.0}}] 
(case type  
       :direct        (OutputFieldDirect. input-field) ;will simply pass the input value to the output (not very useful)
       ;maps a seq of numbers to a specific range. For Support Vector Machines and many neural networks based on the 
       ;HTAN activation function the input must be in the range of -1 to 1. If you are using a sigmoid activation function
       ;you should normalize to the range 0 - 1.
       :range-mapped  (OutputFieldRangeMapped. input-field bottom top)
       ;z-axis should be used when you need a consistent vector length, often for SOMs. Usually a better choice than multiplicative
       :z-axis        (OutputFieldZAxis. group input-field) ;;remember to add a synthetic field in your z-axis group
       ;multiplicative normalisation can be very useful for vector quantization and when you need a consistent vector length. 
       ;It may also perform better than z-axis when all of the input fields are near 0.
       :multiplicative  (OutputFieldMultiplicative. group input-field)
       :encode          (OutputFieldEncode. input-field)
       :nominal     (if one-of-n?
                      (doto (OutputOneOf. top bottom)       ;simplistic one-of-n method (not very good)
                            (.addItem input-field))
                      (doto (OutputEquilateral. top bottom) ;better alternative for nominal values usually
                            (.addItem input-field)))
      (throw (IllegalArgumentException. "Unsupported output-field type!")) 
))


(definline data-normalization [storage] 
`(doto (DataNormalization.) 
   (.setTarget ~storage)))

(defn normalize 
"Function for producing normalised values. It is normally being used from within the main 'prepare' function."
[ins outs max min storage] ;ins must be a seq
(let [norm  (data-normalization storage)]
  (mapv #(do (.addInputField norm %1) (.addOutputField norm %2)) ins outs)
  (.process norm)
(if (every? #(= InputFieldCSV (class %)) ins) 
   (println "SUCCESS...!");there is nothing to return, at least print something
   (.getArray storage)) ))
    

(defn prepare
"Adjusts data to be within a certain range. Defaults to (-1 ... 1). This function does all the setting up needed for normalization. 
 First 3 arguments are mandatory. There rest 5 are optional. 
----------------------------------------------------------------------------------------------------
 inputs  (mandatory -- the InputFields),
 outputs (mandatory -- the OutputFields),
 storage (mandatory -- where to store the normalised values) 
 :how    (optional -- the normalization technique to use; if you provided input/output fields and storage there is no need to use this) 
 Options include:
       :array-range (This is the quickest way to normalize a 1d array within a range without involving input/output fields)  
       :csv-range   (This is what you use for csv files. Use the input file name as 'inputs' and the output file name for 'storage'. 'outputs' is ignored...))
 :top  (optional -- the max value) :default  1
 :bottom    (optional -- the min value) :default -1 
 :raw-seq   (optional -- the raw-seq to normalise in case you're doing :array-range normalisation) *defaults to an empty vector
 :has-headers? (optional -- csv input file includes headers?) *defaults to false." 
[inputs outputs storage & {:keys [how has-headers? top bottom raw-seq] 
                           :or {has-headers? false raw-seq [] top 1.0 bottom -1.0}}] ;;defaults 
(case how  
       :array-range  (let [norm (NormalizeArray.)] ;convenient array normalization
                       (do (.setNormalizedHigh norm top)
                           (.setNormalizedLow norm  bottom)  
                           (.process norm  (double-array raw-seq))))
       :csv-range    ;convenient csv file normalization
                       (let [input  (java.io.File. inputs) ;in this case inputs & storage should be strings (file-names)
                             output (java.io.File. storage)
                             analyst (EncogAnalyst.) 
                             wizard (AnalystWizard. analyst)
                             norm (AnalystNormalizeCSV.)]
                         (. wizard wizard input true AnalystFileFormat/DECPNT_COMMA)
                         (.analyze norm  input has-headers? CSVFormat/ENGLISH analyst)
                         ;(. norm setOutputFormat CSVFormat/ENGLISH)
                         (.setProduceOutputHeaders norm  has-headers?)
                         (.normalize  norm normalize output))                            
  (normalize inputs outputs top bottom storage)))                       


