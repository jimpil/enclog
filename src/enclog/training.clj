(ns enclog.training
(:import
       (org.encog.neural.networks.training CalculateScore)
       (org.encog.neural.networks.training.simple TrainAdaline)
       (org.encog.neural.networks.training.propagation.back  Backpropagation)
       (org.encog.neural.networks.training.propagation.manhattan ManhattanPropagation)              
       (org.encog.neural.networks.training.propagation.quick  QuickPropagation) 
       (org.encog.neural.networks.training.propagation.scg ScaledConjugateGradient)
       (org.encog.neural.networks.training.propagation.resilient ResilientPropagation)
       (org.encog.neural.networks.training.genetic NeuralGeneticAlgorithm)
       (org.encog.neural.networks.training.pnn TrainBasicPNN)
       (org.encog.neural.networks.training.nm NelderMeadTraining)
       (org.encog.neural.networks.training.anneal NeuralSimulatedAnnealing)
       (org.encog.neural.neat.training NEATTraining)
       (org.encog.neural.som.training.basic BasicTrainSOM)
       (org.encog.neural.som.training.basic.neighborhood 
                                    NeighborhoodFunction NeighborhoodRBF NeighborhoodRBF1D 
                                    NeighborhoodBubble NeighborhoodSingle)
       (org.encog.neural.neat NEATPopulation)
       (org.encog.neural.rbf RBFNetwork)
       (org.encog.neural.som SOM) 
       (org.encog.neural.data.basic BasicNeuralDataPair)
       (org.encog.neural.rbf.training SVDTraining)
       (org.encog.ml.data.basic BasicMLData BasicMLDataSet BasicMLDataPair BasicMLSequenceSet)
       (org.encog.ml.data MLDataSet)
       (org.encog.ml.data.temporal TemporalMLDataSet)
       (org.encog.ml.data.folded FoldedDataSet)
       (org.encog.ml.train MLTrain)
       (org.encog.ml.svm.training SVMTrain)
       (org.encog.ml.svm SVM)
       (org.encog.ml MLRegression)
       (org.encog.util.simple EncogUtility)
       (org.encog.util Format)
       (org.encog.neural.networks BasicNetwork)
       (org.encog.mathutil.rbf RBFEnum)
       (org.encog.mathutil.randomize Randomizer BasicRandomizer ConsistentRandomizer 
                                     ConstRandomizer FanInRandomizer Distort GaussianRandomizer
                                     NguyenWidrowRandomizer RangeRandomizer)
       (org.encog.util.arrayutil TemporalWindowArray)
       (org.encog.ml.kmeans KMeansClustering)
       (org.encog.util.text BagOfWords)))


;--------------------------------------*SOURCE*--------------------------------------------------------------------------------
;------------------------------------------------------------------------------------------------------------------------------
(defn data 
"Constructs a MLData object given some data which can be nested. Options include:
 ---------------------------------------------------------------------------------------
 :basic   :basic-dataset  :temporal-window [window-size prediction-size]  
 :basic-complex (not wrapped)  :folded :basic-pair :neural-pair :sequence-set
 ---------------------------------------------------------------------------------------
 Returns the actual MLData object or a closure that needs to be called again with extra arguments." 
[of-type & data]
(case of-type
   :basic   (if (number? (first data)) (BasicMLData. (first data)) ;initialised empty  
                                       (BasicMLData. (double-array (first data)))) ;initialised with train data
   :basic-complex (throw (IllegalArgumentException. "Complex dataset isnot suported at the moment. Consider using encog directly...")) ;;TODO
   :basic-dataset (if (empty? data)  (BasicMLDataSet.) ;initialised empty 
                  (BasicMLDataSet. (into-array (map double-array (first data))) 
                                   (if (nil? (second data)) nil ;there is no ideal data 
                                       (into-array (map double-array (second data))))))
   :temporal-dataset (TemporalMLDataSet. (-> data first int) (-> data second int)) ;;should be numbers
   :temporal-window  (fn [window-size prediction-size]
                       (let [twa (TemporalWindowArray. window-size prediction-size)]
                         (do (.analyze twa (doubles (first data))) 
                             (.process twa (doubles (first data)))))) ;returns array of doubles
   ;:sequence-set expects an entire dataset. So first call 'data' to construct your dataset and then call 'data' again asking for a sequence-set/folded view of it                         
   :sequence-set  (BasicMLSequenceSet. (first data)) 
   :basic-pair  (let [[input ideal & more] data] (if (nil? ideal) (BasicMLDataPair. input) (BasicMLDataPair. input ideal)))
   :neural-pair (let [[input ideal & more] data] (if (nil? ideal) (BasicNeuralDataPair. input) (BasicNeuralDataPair. input ideal)))                                                     
   :folded (FoldedDataSet. (first data)) ;;expects an underlying dataset e.g (->> (data :basic [1 2 3 4 5]) (data :folded))
(throw (IllegalArgumentException. "Unsupported data model!"))
))

(defn neighborhood-F 
"To be used with SOMs. Options for type include:
 -----------------------------------------------
 :single  :bubble  :rbf  :rbf1D
 -----------------------------------------------" 
[type] 
(case type 
       :single (NeighborhoodSingle.) 
       :bubble (fn [radius] (NeighborhoodBubble. (int radius)))
       :rbf    (fn [^RBFEnum t & dims] (NeighborhoodRBF. (int-array dims) t))
       :rbf1D  (fn [^RBFEnum r] (NeighborhoodRBF1D. r))
(throw (IllegalArgumentException. "Unsupported neighborhood type!"))
))

;;example-usage: 
; (neighborhood-F  :single)
; (neighborhood-F  :bubble 5)
; ((neighborhood-F  :rbf1D) (rbf-enum :gaussian))
; ((neighborhood-F  :rbf)   (rbf-enum :mexican-hat) 2 3 5) 

(defn rbf-enum
 "Returns the appropriate RBFEnum given the provided preference. 
 This function is typically used in conjuction with 'make-neighborhoodF' in case 
 we choose rbf/rbf1D neighborhood functions. Generally this will be a :gaussian function.
 Other options include :multiquadric, :inverse-multiquadric, :mexican-hat." 
 [what] 
 (case what
        :gaussian     (RBFEnum/Gaussian)
        :multiquadric (RBFEnum/Multiquadric)
        :inverse-multiquadric  (RBFEnum/InverseMultiquadric)
        :mexican-hat  (RBFEnum/MexicanHat)
 (throw (IllegalArgumentException. "Unsupported rbf-enum!"))
 ))

(defn randomizer 
"Constructs a Randomizer object. Options [with respective args] include:
 ------------------------------------------------
 :range [:min, :max]   
 :consistent [:min, :max]   
 :distort [:factor]
 :constant [:constant] 
 :gaussian [:mean, :st-deviation]  
 :fan-in (if :symmetric? [:boundary,  :sq-root?] 
                         [:min, :max, :sq-root?]) 
 :nguyen-widrow [] 
 ------------------------------------------------
 -examples:
  (randomizer :range :min -1 :max 1)
  (randomizer :distort :factor 0.5)
  (randomizer :constant :constant 0.25)
  (randomizer :fan-in :boundary 0.9 :symmetric? true)
  (randomizer :fan-in :min 0.49 :max 0.9 :sqr-root? true) 
  (randomizer :nguyen-widrow)" 
[type & {:as opts}]
(case type
    :range      (RangeRandomizer. (:min opts) (:max opts)) ;range randomizer
    :consistent (ConsistentRandomizer. (:min opts) (:max opts)) ;consistent range randomizer
    :constant   (ConstRandomizer. (:constant opts))
    :distort    (Distort. (:factor opts))
    :fan-in     (if (:symmetric? opts) (FanInRandomizer. (- (:boundary opts)) (:boundary opts) (boolean (:sqr-root? opts)))
                                       (FanInRandomizer. (:min opts) (:max opts) (boolean (:sqr-root? opts))))
    :gaussian   (GaussianRandomizer. (:mean opts) (:st-deviation opts))
    :nguyen-widrow  (NguyenWidrowRandomizer.) ;the most performant randomizer for networks
(throw (IllegalArgumentException. "Unsupported randomization technique!"))  
))

(defn randomize
"Performs the actual randomization mainly via array mutation. Expects a randomizer object and some data. 
 Pass in vector(s) if you want vector(s) back, otherwise you will get arrays. Data can be:
 MLMethod (network) -- double -- double[] -- double[][] -- Matrix -- clj-vector (1d / 2d).
 Note: Not all randomizers implement randomize() (only FanIn)." 
[^Randomizer randomizer data]
(if-not (vector? data) (do (.randomize randomizer data) data) ;;not a vector (presumably something encog already knows how to handle)
(let [res2 (when (vector? (first data)) (into-array (mapv #(into-array Double/TYPE %) data))) ;;2d vector
      res1 (when-not res2 (double-array data))] ;;1d vector
  (cond res2 (do (.randomize randomizer res2) (vec (map vec res2))) ;got 2d vector return 2d-vectos
        res1 (do (.randomize randomizer res1) (vec res1)) ;got 1d-vector return 1d-vector
  :else (println "NOTHING HAPPENED!!!\n"))))) 
  
  
(defn cluster 
"Simple k-means clustering. Expects raw-data (2d seq), k (the number of clusters to use) and number of iterations.
 Returns a map where keys are numbers from 1 to n clusters we got back and  values are vectors holding the clustered BasicMLData objects." 
[raw-data k iterations]
(let [wrapped (map #(data :basic %) raw-data) ;wrap each inner vector into a BasicMLData object
      dataset (let [ds (data :basic-dataset)] 
                (doseq [el wrapped] (.add ds el)) ds) ;make dataset with no ideal data  
      kmeans  (KMeansClustering. k dataset)  ;the concrete K-Means object  
      ready   (.iteration kmeans iterations) ;how many iterations
      clusters (.getClusters kmeans) ]       ;the actual clusters - an array of MLCluster objects      
 (->> clusters 
   (map (comp vec #(.getData %)))
   (interleave (range 1 (inc k)))
   (apply sorted-map-by >)        
   (merge {:number-of-clusters k})))) 
   
(definline bag-of-words [k]
`(BagOfWords. ~k))    
                             

(definline implement-CalculateScore 
"Consumer convenience for implementing the CalculateScore interface which is needed for genetic and simulated annealing training."
[minimize? eval-fn]
`(reify CalculateScore 
  (^double calculateScore  [this ^MLRegression n#] (~eval-fn n#)) 
  (^boolean shouldMinimize [this] ~minimize?)))
  
(defn add-strategies [^MLTrain method & strategies]
"Consumer convenience for adding strategies to a training method. Returns the modified training method."
(doseq [s strategies]
  (.addStrategy method s)) method)  

(defn trainer
"Constructs a training-method (MLTrain) object  given a method. Options [with respective args] inlude:
 -------------------------------------------------------------
 :simple-adaline [:network, :training-set, :learning-rate]      
 :quick-prop [:network, :training-set, :learning-rate]     
 :manhattan  [:network, :training-set, :learning-rate]
 :back-prop      [:network, training-set]
 :resilient-prop [:network, training-set]   
 :neat   [:score-fn,  :minimize?, :input, :output, :population-size/:population-object]     
 :genetic [:network, :randomizer, :fitness-fn, :minimize?, population-size :mutation-percent, :mate-percent]   
 :svm  [:network, :training-set]
 :pnn  [:network, :training-set]
 :basic-som   [:network, :training-set, :learning-rate, :neighborhood-fn]      
 :nelder-mead [:network, :training-set, :step]    
 :annealing   [:network, :fitness-fn, :minimize?, :start-temperature, :stop-temperature, :cycles]  
 :svd  [:network, :training-set] 
 :scaled-conjugent [:network, :training-set]                          
 -------------------------------------------------------------"  
[method & {:as opts}]
(case method
       :simple-adaline (TrainAdaline. (:network opts) (:training-set opts) (:learning-rate opts))
       :back-prop  (Backpropagation.  (:network opts) (:training-set opts)) 
       :manhattan  (ManhattanPropagation. (:network opts) (:training-set opts) (:learning-rate opts))
       :quick-prop (QuickPropagation. (:network opts) (:training-set opts) (if-let [lr (:learning-rate opts)] lr 2.0))
       :genetic    (NeuralGeneticAlgorithm. (:network opts) (:randomizer opts) 
                                            (implement-CalculateScore (:minimize? opts) (:fitness-fn opts)) 
                                            (:population-size opts) (:mutation-percent opts) (:mate-percent opts))
       :scaled-conjugent   (ScaledConjugateGradient. (:network opts) (:training-set opts))
       :pnn                (TrainBasicPNN.  (:network opts) (:training-set opts))
       :annealing         (NeuralSimulatedAnnealing. (:network opts) 
                          (implement-CalculateScore (:minimize? opts) (:fitness-fn opts)) 
                              (:start-temperature opts) (:stop-temperature opts)  (:cycles opts))
       :resilient-prop (ResilientPropagation. (:network opts) (:training-set opts))
       :nelder-mead    (NelderMeadTraining. (:network opts)  (if-let [st (:step opts)] st 100))
       :svm            (SVMTrain. (:network opts) (:training-set opts))
       :svd            (SVDTraining. (:network opts) (:training-set opts))
       :basic-som      (BasicTrainSOM. (:network opts) (:learning-rate opts) (:training-set opts) (:neighborhood-fn opts))
       :neat       (if-not (:population-object opts)
       ;;neat creates a population so we don't really need an actual network. We can skip the 'make-network' bit.
       ;;population can be an integer or a NEATPopulation object 
                          (NEATTraining. (implement-CalculateScore (:minimize? opts) (:fitness-fn opts)) 
                                         (:input opts) (:output opts) (:population-size opts))
                          (NEATTraining. (implement-CalculateScore (:minimize? opts) (:fitness-fn opts)) 
                                         (:population-object opts))) 
 (throw (IllegalArgumentException. "Unsupported training method!"))     
))


;;usage: (train (trainer :resilient-prop :network (network blah-blah) some-data-set)  0.001 500 [])
                
                               
(defn train 
"Does the actual training. This is a potentially lengthy and costly process. 
 This is an overloaded fucntion. It is up to you whether you want to provide limits for error-tolerance (pass Double/NEGATIVE_INFINITY if you don't care), 
 iteration-number or both. Regardless of the limitations however, this  function will return the best network so far
 as a result of training."
([^MLTrain method error-tolerance limit strategies] ;;eg: (new RequiredImprovementStrategy 5) 
 (apply add-strategies method strategies) 
    (loop [epoch (int 1)]
      (if (< limit epoch) (.getMethod method) ;failed to converge - return the best network
       (do (.iteration method)
           (println "Iteration #" (Format/formatInteger epoch) 
                    "Error:" (Format/formatPercent (.getError method)) 
                    "Target-Error:" (Format/formatPercent error-tolerance))
       (if (< (.getError method) error-tolerance) (.getMethod method) ;;succeeded to converge -return the best network 
       (recur (inc epoch)))))))
([^MLTrain method error-tolerance strategies] 
  (apply add-strategies method strategies)
  (EncogUtility/trainToError method error-tolerance)
  (.getMethod method))
([^MLTrain method strategies] ;;need only one iteration - SVMs or Nelder-Mead training for example
(apply add-strategies method strategies)
  (.iteration method) 
  (println "Error:" (.getError method)) 
  (.getMethod method)))

(definline evaluate 
"This expands to EncogUtility.evaluate(n,d). Expects a network and a dataset and prints the evaluation." 
[n ds]
 `(EncogUtility/evaluate ~n ~ds)) 


 
