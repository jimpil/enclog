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
       (org.encog.util.normalize.output OutputFieldRangeMapped)
       (org.encog.util.normalize.output.multiplicative OutputFieldMultiplicative 
                                                       MultiplicativeGroup)
       (org.encog.util.normalize.output.zaxis OutputFieldZAxis ZAxisGroup)
       (org.encog.util.normalize.output OutputFieldDirect)  
       (org.encog.app.analyst EncogAnalyst AnalystFileFormat)
       (org.encog.app.analyst.wizard AnalystWizard)
       (org.encog.app.analyst.csv.normalize AnalystNormalizeCSV)                                 
       (org.encog.util.csv CSVFormat )                                
       (org.encog.util.normalize.output.nominal OutputEquilateral OutputOneOf)                               
       (org.encog.util.arrayutil NormalizeArray)
))
;--------------------------------------------   
(defn target-storage
"Constructs a Normalization storage facility. Options include:" 
[type & sizes]
(condp = type
       :norm-array     (NormalizationStorageArray1D. (make-array Double/TYPE (first sizes))) ;just rows
       :norm-array2d   (NormalizationStorageArray2D. (make-array Double/TYPE (first sizes) (second sizes))) ;columns,rows
       :norm-csv   (fn [^String filename] (NormalizationStorageCSV. (java.io.File. filename))) ;where to write the csv file
       :norm-dataset   (NormalizationStorageNeuralDataSet. (first sizes) (second sizes)) ;input-count & ideal-count
:else (throw (IllegalArgumentException. "Unsupported storage-target type!"))
))    
 
    
(defn input
"Constructs an input field to be used with the DataNormalization class. Options include:
----------------------------------------------------------------------------------------
:basic   :csv   :array-1d    :array-2d  
----------------------------------------------------------------------------------------" 
[element &{:keys [forNetwork? type column-offset index2] 
           :or {forNetwork? true type :array-1d column-offset 5}}] 
(condp = type
         :basic     (fn [] (let [inf (BasicInputField.)] 
                            (do (. inf setCurrentValue element) inf)))  ;element must be a Number
         :csv    (fn [] (InputFieldCSV. forNetwork? element column-offset)) ;element must be a java.io.File     
         :array-1d  (fn [] (InputFieldArray1D. forNetwork? (double-array element))) ; element must be a seq     
         :array-2d  (fn [index2] 
                        (InputFieldArray2D. forNetwork? 
                              (into-array (map double-array element)) index2)) ; element must be a 2d seq
                 :else (throw (IllegalArgumentException. "Unsupported input-field type!"))    
 ))
 
(defn output
"Constructs an output field to be used with the DataNormalization class. Options include:
----------------------------------------------------------------------------------------
:direct  :range-mapped  :z-axis   :multiplicative  :nominal 
----------------------------------------------------------------------------------------" 
[input-field &{:keys [forNetwork? type]
                       :or {type :range-mapped}}] 
(condp = type  
       ;will simply pass the input value to the output (not very useful)
       :direct   (fn [] 
                   (OutputFieldDirect. input-field))
       :range-mapped  (fn [low high] 
                          (OutputFieldRangeMapped. input-field low high))
       :z-axis   (fn [group] 
                   (OutputFieldZAxis. (ZAxisGroup.) input-field))
       :multiplicative (fn [group] 
                           (OutputFieldMultiplicative. 
                           (MultiplicativeGroup.) input-field))
       :nominal    (fn [one-of-n? high low]
                    (if one-of-n?
                      (doto (OutputOneOf. high low)       ;simplistic one-of-n method (not very good)
                            (.addItem input-field))
                      (doto (OutputEquilateral. low high) ;better alternative for nominal values usually
                            (.addItem input-field))))
      :else (throw (IllegalArgumentException. "Unsupported output-field type!")) 
))


                         


(defmacro make-data-normalization [storage] 
`(let [dn# (DataNormalization.)]
 (do (. dn# setTarget ~storage) dn#)))

(defn normalize "Function for producing normalised values. It is normally being used from within the main normalize function."
[how ins outs max min , batch? storage] ;ins must be a seq
(let [norm  (make-data-normalization storage)]
(do  (dotimes [i (count ins)] 
       (. norm addInputField  (nth ins i))
       (. norm addOutputField (nth outs i)))                
    (. norm process) 
    
    (if (every? #(= InputFieldCSV (class %)) ins) 
        (println "SUCCESS...!");there is nothing to return
    (.getArray storage)))))
    
 ;(if (every? (= InputFieldCSV (class (first ins))) (println "SUCCESS...!")  ;;Not elegant
 ;(.getArray storage))))) ;returns the normalised array found into the storage target
  
   
 ;(do  #_(println (seq (second source) ))
     ; (. norm addInputField  in)          
     ; (. norm addOutputField out)
     ; (. norm process) 
      ;(.getArray storage)))) ;returns the normalised array found into the storage target

(defn prepare
"Adjusts data to be within a certain range. Defaults to (-1 ... 1). This function does all the setting up needed for normalization. First 3 arguments are mandatory. There 4 optional ones as well. 
----------------------------------------------------------------------------------------------------
 :how    (mandatory -- the normalization technique to use) 
 Options include:
              :array-range (This is the quickest way to normalize a 1d array within a range)  
              :csv-range   (This is what you use if you can't get rid of headers from csv file.))
              :range  (This involves creating proper input/output fields for columns of either a 1d or a 2d structure))
              :z-axis (This is good for  consistent vector lengths, often for SOMs. Usually a better choice than multiplicative)
              :multiplicative 
 inputs  (mandatory -- the InputFields),
 outputs (mandatory -- the OutputFields), 
 :forNetwork? (optional -- are we normalizing for network input?)  *defaults to true
 :ceiling  (optional -- the max value) :default  1
 :floor    (optional -- the min value) :default -1 ." 
[how inputs outputs &{:keys [forNetwork? ceiling floor raw-seq]  ;;4 keys
             :or {forNetwork? true ceiling 1.0 floor -1.0}}] ;;defaults 
(condp = how 
       :array-range  (let [norm (NormalizeArray.)] ;convenient array normalization
                       (do (. norm setNormalizedHigh ceiling)
                           (. norm setNormalizedLow  floor)  
                           (. norm process (double-array raw-seq))))
       :csv-range    (fn [source-file target-file  has-headers?] ;convenient csv file normalization
                       (let [input  (java.io.File. source-file)
                             output (java.io.File. target-file)
                             analyst (EncogAnalyst.) 
                             wizard (AnalystWizard. analyst)
                             norm (AnalystNormalizeCSV.)]
                        (do (. wizard wizard input true AnalystFileFormat/DECPNT_COMMA)
                            (. norm analyze input has-headers? CSVFormat/ENGLISH analyst)
                            ;(. norm setOutputFormat CSVFormat/ENGLISH)
                            (. norm setProduceOutputHeaders has-headers?)
                            (. norm normalize output))))     
                             
                             
       ;maps a seq of numbers to a specific range. For Support Vector Machines and many neural networks based on the 
       ;HTAN activation function the input must be in the range of -1 to 1. If you are using a sigmoid activation function
       ;you should normalize to the range 0 - 1.
       :range          (partial normalize :range-mapped inputs outputs ceiling floor)   
       ;z-axis should be used when you need a consistent vector length, often for SOMs. Usually a better choice than multiplicative
       :z-axis         (partial normalize :z-axis inputs outputs ceiling floor)
       ;multiplicative normalisation can be very useful for vector quantization and when you need a consistent vector length. 
       ;It may also perform better than z-axis when all of the input fields are near 0.
       :multiplicative (partial normalize :multiplicative inputs outputs ceiling floor)
       ;reciprocal normalization is always normalizing to a number in the range between 0 and 1. Very simple technique.
       :reciprocal  nil ;TODO 
))                       


