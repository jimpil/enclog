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
       (org.encog.neural.rbf.training SVDTraining)
       (org.encog.ml.data.basic BasicMLData BasicMLDataSet)
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
       (org.encog.util.arrayutil TemporalWindowArray)))


;--------------------------------------*SOURCE*--------------------------------------------------------------------------------
;------------------------------------------------------------------------------------------------------------------------------
(defn data 
"Constructs a MLData object given some data which can be nested as well. Options include:
 ---------------------------------------------------------------------------------------
 :basic   :basic-dataset  :temporal-window  :basic-complex (not wrapped) 
 :folded (not wrapped) 
 Returns the actual MLData object or a closure that needs to be called again with extra arguments." 
[of-type & data]
(condp = of-type
   :basic         (if (number? data) (BasicMLData. (count (first data))) ;initialised empty  
                                     (BasicMLData. (double-array (first data)))) ;initialised with train data
   :basic-complex nil ;;TODO
   :basic-dataset (if (nil? data)  (BasicMLDataSet.) ;initialised empty 
                  (BasicMLDataSet. (into-array (map double-array (first data))) 
                                   (if (nil? (second data)) nil ;there is no ideal data 
                                       (into-array (map double-array (second data))))))
   ;:temporal-dataset (TemporalMLDataSet. ) 
   :temporal-window (fn [window-size prediction-size]
                           (let [twa (TemporalWindowArray. window-size prediction-size)]
                           (do (. twa analyze (doubles (first data))) 
                               (. twa process (doubles (first data)))))) ;returns array of doubles
                           
                                                          
   ;:folded (FoldedDataSet.)
:else (throw (IllegalArgumentException. "Unsupported data model!"))
))

(defn make-neighborhoodF 
"To be used with SOMs. Options for type include:
 -----------------------------------------------
 :single  :bubble  :rbf  :rbf1D
 -----------------------------------------------" 
[type] 
(condp = type 
       :single (NeighborhoodSingle.) 
       :bubble (fn [radius] (NeighborhoodBubble. (int radius)))
       :rbf    (fn [^RBFEnum t & dims] (NeighborhoodRBF. (int-array dims) t))
       :rbf1D  (fn [^RBFEnum r] (NeighborhoodRBF1D. r))
))

;;example-usage: 
; (make-neighborhoodF :single)
; (make-neighborhoodF :bubble 5)
; ((make-neighborhoodF :rbf1D) (make-RBFEnum :gaussian))
; ((make-neighborhoodF :rbf)   (make-RBFEnum :mexican-hat) 2 3 5) 

(defn make-RBFEnum
 "Returns the appropriate RBFEnum given the provided preference. 
 This function is typically used in conjuction with 'make-neighborhoodF' in case 
 we choose rbf/rbf1D neighborhood functions. Generally this will be a :gaussian function.
 Other options include :multiquadric, :inverse-multiquadric, :mexican-hat." 
 [what] 
 (condp = what
        :gaussian     (RBFEnum/Gaussian)
        :multiquadric (RBFEnum/Multiquadric)
        :inverse-multiquadric  (RBFEnum/InverseMultiquadric)
        :mexican-hat  (RBFEnum/MexicanHat)
 ))

(defn randomizer 
"Constructs a Randomizer object. Options include:
 ------------------------------------------------
 :basic    :range   :consistent   :distort  
 :constant :fan-in  :gaussian     :nguyen-widrow 
 ------------------------------------------------
 Returns a Randomizer object or a closure." 
[type]
(condp = type
    :basic      (BasicRandomizer.) ;random number generator with a random(current time) seed
    :range      (fn [min-val max-val] (RangeRandomizer. min-val max-val)) ;range randomizer
    :consistent (fn [min-rand max-rand] (ConsistentRandomizer. min-rand max-rand)) ;consistent range randomizer
    :constant   (fn [constant] (ConstRandomizer. constant))
    :distort    (fn [factor]   (Distort. factor))
    :fan-in     (fn [boundary sqr-root?] (FanInRandomizer. (- boundary) boundary (if (nil? sqr-root?) false sqr-root?)))
    :gaussian   (fn [mean st-deviation] (GaussianRandomizer. mean st-deviation))
    :nguyen-widrow  (NguyenWidrowRandomizer.) ;the most performant randomizer  
))

(defn randomize
"Performs the actual randomization. Expects a randomizer object and some data. Options for data include:
 MLMethod -- double -- double[] -- double[][] -- Matrix " 
[^Randomizer randomizer data] 
(. randomizer randomize data))

(defmacro implement-CalculateScore 
"Consumer convenience for implementing the CalculateScore interface which is needed for genetic and simulated annealing training."
[minimize? & body]
`(reify CalculateScore 
  (^double calculateScore  [this ^MLRegression n#] ~@body) 
  (^boolean shouldMinimize [this] ~minimize?)))
  
(defmacro add-strategies [^MLTrain method & strategies]
"Consumer convenience for adding strategies to a training method."
`(doseq [s# ~@strategies]
 (.addStrategy ~method s#)))  

(defn trainer
"Constructs a training-method object given a method. Options inlude:
 -------------------------------------------------------------
 :simple     :back-prop    :quick-prop      :manhattan   :neat        
 :genetic    :svm          :nelder-mead     :annealing   :svd  
 :scaled-conjugent         :resilient-prop  :pnn                        
 ------------------------------------------------------------- 
 Returns a MLTrain object."
[method]
(condp = method
       :simple     (fn [net tr-set learn-rate] (TrainAdaline.  net tr-set (if (nil? learn-rate) 2.0 learn-rate)))
       :back-prop  (fn [net tr-set] (Backpropagation. net tr-set))
       :manhattan  (fn [net tr-set learn-rate] (ManhattanPropagation. net tr-set learn-rate))
       :quick-prop (fn [net tr-set learn-rate] (QuickPropagation. net tr-set (if (nil? learn-rate) 2.0 learn-rate)))
       :genetic    (fn [net randomizer fit-fun minimize? pop-size mutation-percent mate-percent] 
                       (NeuralGeneticAlgorithm. net randomizer 
                                                   (implement-CalculateScore minimize? fit-fun) 
                                                    pop-size mutation-percent mate-percent))
       :scaled-conjugent   (fn [net tr-set] (ScaledConjugateGradient. net tr-set))
       :pnn                (fn [net tr-set] (TrainBasicPNN. net tr-set))
       :annealing     (fn [net fit-fun minimize? startTemp stopTemp cycles] 
                          (NeuralSimulatedAnnealing. net 
                          (implement-CalculateScore minimize? fit-fun) startTemp stopTemp cycles))
       :resilient-prop (fn [net tr-set]      (ResilientPropagation. net tr-set))
       :nelder-mead    (fn [net tr-set step] (NelderMeadTraining. net tr-set (if (nil? step) 100 step)))
       :svm            (fn [^SVM net tr-set] (SVMTrain. net tr-set))
       :svd            (fn [^RBFNetwork net ^MLDataSet tr-set] (SVDTraining. net tr-set))
       :basic-som      (fn [^SOM net learn-rate ^MLDataSet tr-set ^NeighborhoodFunction neighborhood] 
                            (BasicTrainSOM. net learn-rate tr-set neighborhood))
       :neat       (fn ([score-fun minimize? input output ^Integer population-size]
       ;;neat creates a population so we don't really need an actual network. We can skip the 'make-network' bit.
       ;;population can be an integer or a NEATPopulation object 
                          (NEATTraining. (implement-CalculateScore minimize? score-fun) input output population-size))
                       ([score-fun minimize? ^NEATPopulation population] 
                          (NEATTraining. (implement-CalculateScore minimize? score-fun) population))) 
 :else (throw (IllegalArgumentException. "Unsupported training method!"))      
))


;;usage: ((make-trainer :resilient-prop) (make-network blah-blah) some-data-set)
;;       ((make-trainer :genetic) (make-network blah-blah) some-data-set)


(defmacro genericTrainer [method & args]
`(fn [& details#] 
   (new ~method (first ~@args) ;the network
                (second ~@args);the training set 
                (rest (rest ~@args)))))
                
                               
(defn train 
"Does the actual training. This is a potentially lengthy and costly process. Returns true or false depending on whether the error target was met within the iteration limit. This is an overloaded fucntion. It is up to you whether you want to provide limits for error-tolerance, iteration-number or both. Regardless of the limitations however, this  functions will always return the best network so far."
([^MLTrain method error-tolerance limit strategies] ;;eg: (new RequiredImprovementStrategy 5) 
(when (seq strategies) (dotimes [i (count strategies)] 
                       (.addStrategy method (get strategies i))))
     (loop [epoch (int 1)]
       (if (< limit epoch) (.getMethod method) ;failed to converge - return the best network
       (do (.iteration method)
           (println "Iteration #" (Format/formatInteger epoch) 
                    "Error:" (Format/formatPercent (. method getError)) 
                    "Target-Error:" (Format/formatPercent error-tolerance))
       (if-not (> (.getError method) error-tolerance) (.getMethod method) ;;succeeded to converge -return the best network 
       (recur (inc epoch)))))))

([^MLTrain method error-tolerance strategies] 
(when (seq strategies) (dotimes [i (count strategies)] 
                       (.addStrategy method (get strategies i))))
(do (EncogUtility/trainToError method error-tolerance)
                                (. method getMethod)))

([^MLTrain method strategies] ;;need only one iteration - SVMs or Nelder-Mead training for example
 (when (seq strategies) (dotimes [i (count strategies)] 
                        (.addStrategy method (get strategies i))))
     (do (.iteration method) 
         (println "Error:" (.getError method)) 
         (.getMethod method))))

(defmacro evaluate 
"This expands to EncogUtility.evaluate(n,d). Expects a network and a dataset and prints the evaluation." 
[n ds] `(EncogUtility/evaluate ~n ~ds)) 




 
