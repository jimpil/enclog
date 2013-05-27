(ns enclog.nnets
(:import 
[org.encog.neural.pattern 
       FeedForwardPattern ADALINEPattern ART1Pattern BAMPattern
       BoltzmannPattern CPNPattern ElmanPattern HopfieldPattern PatternError
       JordanPattern SOMPattern PNNPattern SVMPattern RadialBasisPattern]
       
[org.encog.engine.network.activation 
       ActivationTANH ActivationSigmoid ActivationGaussian ActivationBiPolar 
       ActivationLinear  ActivationLOG ActivationRamp ActivationSoftMax ActivationStep
       ActivationSIN ActivationBipolarSteepenedSigmoid ActivationClippedLinear
       ActivationCompetitive ActivationElliott ActivationElliottSymmetric ActivationSteepenedSigmoid]     
[org.encog.ml.bayesian BayesianNetwork BayesianEvent]    
   ))


 (defn neural-pattern  
 "Constructs a neural-pattern from which we can generate 
 a concrete network effortlessly (see multi-fn 'network' for more).
 Options include:
 -------------------------------------------------------------
 :feed-forward  :adaline  :art1  :bam  :boltzman      
 :jordan        :elman    :svm   :rbf  :hopfield    
 :som           :pnn      :cpn   :bayesian (returns keyword)               
 -------------------------------------------------------------
 Returns an object that implements NeuralNetworkPattern."
  [model & {:as opts}]
  (case model
       :feed-forward (FeedForwardPattern.)
       :adaline      (ADALINEPattern.)
       :art1         (ART1Pattern.)
       :bam          (BAMPattern.)
       :boltzman     (BoltzmannPattern.)
       :cpn          (CPNPattern.)
       :elman        (ElmanPattern.)
       :hopfield     (HopfieldPattern.)
       :jordan       (JordanPattern.)
       :som          (SOMPattern.)
       :pnn          (PNNPattern.)
       :bayesian     :bayesian ;;there is no pattern to instantiate - this is useless - call network instead
       :svm          (if-not (:regression? opts) (SVMPattern.)
                            (doto (SVMPattern.) (.setRegression true)))
       :rbf          (RadialBasisPattern.)
   (throw (IllegalArgumentException. "Unsupported neural-pattern!"))
 ))

;;usage: 
;;(make-pattern :feed-forward)
;;(make-pattern :svm)

 (defn activation  
 "Constructs an activation-function to be used by the nodes of the network.
  Expects a keyword argument. Options include:
  --------------------------------------------------
  :tanh     :sigmoid   :gaussian    :bipolar  
  :linear   :log       :ramp        :sin
  :elliot   :soft-max  :competitive :bipolar-steepend
  :elliot-symmetric :clipped-linear :steepened-sigmoid
  ---------------------------------------------------
  Returns an ActivationFunction object." ;TODO
 [fun]
 (case fun 
      :tanh     (ActivationTANH.)
      :sigmoid  (ActivationSigmoid.)
      :gaussian (ActivationGaussian.)
      :bipolar  (ActivationBiPolar.)
      :linear   (ActivationLinear.) 
      :log      (ActivationLOG.)
      :ramp     (ActivationRamp.)
      :sin      (ActivationSIN.)
      :soft-max (ActivationSoftMax.)
      :step     (ActivationStep.)
      :bipolar-steepend (ActivationBipolarSteepenedSigmoid.)
      :clipped-linear   (ActivationClippedLinear.)
      :competitive      (ActivationCompetitive.)
      :elliot           (ActivationElliott.)
      :elliot-symmetric (ActivationElliottSymmetric.)
      :steepened-sigmoid (ActivationSteepenedSigmoid.)
   (throw (IllegalArgumentException. "Unsupported activation-function!"))
 ))
 
 (defn bayesian-event [label & choices]
 (if (empty? choices) 
    (BayesianEvent. label)
    (BayesianEvent. label (into-array choices))))


(defmulti network 
"Depending on the neural-pattern passed in, constructs the appropriate network with some layers and an activation-function.
 Options do not need to be in a  map literal. Some networks do not accept hidden layers or a settable activation 
 but you don't worry if youpass in anything weird - it will be ignored. 
 SVM & PNN networks can take a couple of extra args, apart from :input and :output in case you don't want defaults. 
 See examples below.
 ------------------------------------------------------------------------------------------------------------ 
 -examples: 
  (network  (neural-pattern :feed-forward)  ;;the classic feed-forward pattern
            :activation :tanh              ;;hyperbolic tangent for activation function
            :input 32 
            :output 1 
            :hidden [40 10 5])              ;;3 hidden layers (first has 40 neurons, second 10, third 5)
            
  (network  (neural-pattern :svm)       ;;a support-vector-machine
             :input 10
             :svm-type  -choices:
                        [org.encog.ml.svm.SVMType/EpsilonSupportVectorRegression (default)
                         org.encog.ml.svm.SVMType/NewSupportVectorClassification 
                         org.encog.ml.svm.SVMType/NewSupportVectorRegression 
                         org.encog.ml.svm.SVMType/SupportVectorClassification
                         org.encog.ml.svm.SVMType/SupportVectorOneClass] 
     
             :kernel-type -choices: 
                          [org.encog.ml.svm.KernelType/Linear
                           org.encog.ml.svm.KernelType/Poly
                           org.encog.ml.svm.KernelType/Precomputed
                           org.encog.ml.svm.KernelType/RadialBasisFunction (default) 
                           org.encog.ml.svm.KernelType/Sigmoid]  ) 
             
  (network  (neural-pattern :pnn) 
             :input 20
             :output 5
             :kernel    -choices:
                       [org.encog.neural.pnn.PNNKernelType/Gaussian  (default)
                        org.encog.neural.pnn.PNNKernelType/Reciprocal ]
        
             :out-model -choices:
                        [org.encog.neural.pnn.PNNOutputMode/Classification
                         org.encog.neural.pnn.PNNOutputMode/Regression  (default)
                         org.encog.neural.pnn.PNNOutputMode/Unsupervised] )           
--------------------------------------------------------------------------------------------------------
Returns the complete neural-network object with randomized weights."
(fn [pattern & layers] (class pattern)));;dispatch-fn only cares about the class of the neural-pattern

(defmethod network FeedForwardPattern
[pattern & {:as layers}] 
    (do 
       (.setInputNeurons pattern  (:input layers))
       (.setOutputNeurons pattern (:output layers))
       (.setActivationFunction pattern (activation (:activation layers)));many activations are allowed with this topology
(doseq [neuron-number (:hidden layers)] 
  (.addHiddenLayer pattern neuron-number))
(.generate pattern)))  ;;finally, return the complete network object
 
(defmethod network ADALINEPattern ;no hidden layers - only ActivationLinear
[pattern & {:as layers}] ;;ignoring activation 
 (->  (doto pattern  
         (.setInputNeurons  (:input layers))   
         (.setOutputNeurons (:output layers)))
  .generate)) 


(defmethod network ART1Pattern ;;no hidden layers - only ActivationLinear
[pattern & {:as layers}] ;;ignoring activation 
  (-> (doto pattern  
         (.setInputNeurons  (:input layers))
         (.setOutputNeurons (:output layers)))
  .generate))

(defmethod network BAMPattern ;no hidden layers - only ActivationBiPolar
[pattern & {:as layers}] ;;ignoring layers and activation 
 (-> (doto pattern  
         (.setF1Neurons (:input layers))
         (.setF2Neurons (:output layers)))
   .generate))
 
(defmethod network BoltzmannPattern ;;no hidden layers - only ActivationBipolar
[pattern _ & {:as layers}] ;;ignoring activation 
 (-> (doto pattern
      (.setInputNeurons (:input layers)))
  .generate))
 
(defmethod network CPNPattern ;;one hidden layer - only ActivationBipolar + Competitive
[pattern & {:as layers}] ;;ignoring activation 
 (-> (doto pattern  
         (.setInputNeurons  (:input layers))
         (.setInstarCount   (get (:hidden layers) 0))
         (.setOutputNeurons (:output layers)))
  .generate))
 
(defmethod network ElmanPattern;;one hidden layer only - settable activation
[pattern & {:as layers}] 
  (-> (doto pattern  
         (.setInputNeurons  (:input layers))
         (.setOutputNeurons (:output layers))
         (.addHiddenLayer (get (:hidden layers) 0))
         (.setActivationFunction (activation (:activation layers)))) 
   .generate))


(defmethod network HopfieldPattern ;;one hidden layer only - settable activation
[pattern & {:as layers}] 
 (-> (doto pattern 
         (.setInputNeurons  (:input layers))
         (.setOutputNeurons (:output layers))
         (.addHiddenLayer   (get (:hidden layers) 0))
         (.setActivationFunction (activation (:activation layers))))
  .generate))  
 
 
(defmethod network JordanPattern ;;one hidden layer only - settable activation
[pattern & {:as layers}] 
  (-> (doto pattern 
         (.setInputNeurons (:input layers))
         (.setOutputNeurons (:output layers))
         (.addHiddenLayer  (get (:hidden layers) 0))
         (.setActivationFunction pattern (activation (:activation layers))))
  .generate))


(defmethod network SOMPattern ;non settable activation - no hidden layers
[pattern & {:as layers}] ;;ignoring activation
  (-> (doto pattern  
         (.setInputNeurons  (:input layers))
         (.setOutputNeurons (:output layers)))
   .generate))
 
(defmethod network RadialBasisPattern ;usually has one hidden layer - non settable activation
[pattern & {:as layers}] ;;ignoring activation
  (-> (doto pattern  
         (.setInputNeurons  (:input layers))
         (.setOutputNeurons (:output layers))
         (.addHiddenLayer (get (:hidden layers) 0)))
  .generate))
    
    
(defmethod network SVMPattern ;;no hidden layers - non settable activation
[pattern & {:as opts}] ;;ignoring activation - only bi-polar is allowed
 (let [tempsvm    (:svm-type opts)
       tempkernel (:kernel-type opts)
       svm-type    (if (nil? tempsvm) org.encog.ml.svm.SVMType/EpsilonSupportVectorRegression tempsvm) 
       kernel-type (if (nil? tempkernel) org.encog.ml.svm.KernelType/RadialBasisFunction tempkernel)] 
  (-> (doto pattern 
         (.setInputNeurons (:input opts))
         (.setOutputNeurons 1) ;only one output is allowed
         (.setKernelType kernel-type)
         (.setSVMType svm-type)) 
  .generate)))
    
(defmethod network PNNPattern ;;no hidden layers - only LinearActivation 
[pattern & {:as opts}] ;;ignoring activation - only LinearActivation is allowed
(let [tempk    (:kernel opts)
      tempout  (:out-model opts)
      kernel    (if (nil? tempk)   org.encog.neural.pnn.PNNKernelType/Gaussian tempk) 
      out-model (if (nil? tempout) org.encog.neural.pnn.PNNOutputMode/Regression  tempout)]
 (->  (doto pattern 
         (.setInputNeurons  (:input opts))
         (.setOutputNeurons (:output opts))
         (.setKernel kernel)
         (.setOutmodel out-model)) 
  .generate))) 
  
  
(defmethod network clojure.lang.Keyword
[pattern & {:as opts}]
(case pattern
    :bayesian 
      (let [n (BayesianNetwork.)]
        (if-not (nil? opts) 
        (do 
          (doseq [[label & choices] (:events opts)] 
               (.createEvent n (apply bayesian-event label choices)))
                  (doseq [[e1 e2] (partition 2  (:dependencies opts))]
                    (.createDependency n (.get (.getEventMap n) e1) 
                                               (.get (.getEventMap n) e2)))
  (doto n (.finalizeStructure))) n)) ;network ready!
))    
;-----------------------------------------------------------------------------------    

(defmethod network :default 
[_ _] ;;ignoring everything
(throw (IllegalArgumentException. "Pattern not recognised or not supported!")))
