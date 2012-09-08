(ns enclog.nnets
(:import 
(org.encog.neural.pattern 
       FeedForwardPattern ADALINEPattern ART1Pattern BAMPattern
       BoltzmannPattern CPNPattern ElmanPattern HopfieldPattern PatternError
       JordanPattern SOMPattern PNNPattern SVMPattern RadialBasisPattern)
       
(org.encog.engine.network.activation 
       ActivationTANH ActivationSigmoid ActivationGaussian ActivationBiPolar 
       ActivationLinear  ActivationLOG ActivationRamp ActivationSoftMax ActivationStep
       ActivationSIN ActivationBipolarSteepenedSigmoid ActivationClippedLinear
       ActivationCompetitive ActivationElliott ActivationElliottSymmetric ActivationSteepenedSigmoid)     
    
   ))


 (defn neural-pattern  
 "Constructs a neural base pattern from which we can generate 
 a concrete network effortlessly (see network for details).
 Options include:
 -------------------------------------------------------------
 :feed-forward  :adaline  :art1  :bam  :boltzman      
 :jordan        :elman    :svm   :rbf  :hopfield    
 :som           :pnn      :cpn                  
 -------------------------------------------------------------
 Returns an object that implements NeuralNetworkPattern."
  [model]
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
       :svm          (SVMPattern.)
       :rbf          (RadialBasisPattern.)
 ;:else (throw (IllegalArgumentException. "Unsupported neural-pattern!"))
 ))

;;usage: 
;;(make-pattern :feed-forward)
;;(make-pattern :svm)

 (defn activation  
 "Constructs an activation-function to be used by the layers.
  Expects a keyword based argument. Options include:
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
 ;:else (throw (IllegalArgumentException. "Unsupported activation-function!"))
 ))

;;usage: (make-activationF :tanh)
;;       (make-activationF :linear)


(defmulti network 
"Depending on the neural-pattern, constructs a neural-network with some layers and an activation. layers do not need to be in a  map literal. Some networks do not accept hidden layers or a settable activation but you shouldn't worry-the parameter will be ignored. See example usage below.
 Returns the complete neural-network object with randomized weights or in case of SVMs and PNNs a function, that needs to be called again (potentially with nil arguments for default) in order to produce the actual SVM/PNN object.
 example: 
  (network (activation :tanh)               ;;hyperbolic tangent for activation function
           (neural-pattern :feed-forward))  ;;the classic feed-forward pattern
            :input 32 
            :output 1 
            :hidden [40, 10, 5]   ;;3 hidden layers (first has 40 neurons, second 10, third 5) ."
(fn [pattern _  & layers] (class pattern)));;dispatch fn only cares about the class of the pattern

(defmethod network FeedForwardPattern
[pattern activation & {:as layers}] 
    (do 
       (.setInputNeurons pattern  (:input layers))
       (.setOutputNeurons pattern (:output layers))
       (.setActivationFunction pattern activation);many activations are allowed here
 (dotimes [i (count (:hidden layers))] 
         (.addHiddenLayer pattern ((:hidden layers) i)))
         (.generate pattern)))  ;;finally, return the complete network object
 
(defmethod network ADALINEPattern ;no hidden layers - only ActivationLinear
[pattern _ & {:as layers}] ;;ignoring activation 
    (doto pattern  
         (.setInputNeurons  (:input layers))   
         (.setOutputNeurons (:output layers)))
    (.generate pattern)) 


(defmethod network ART1Pattern ;;no hidden layers - only ActivationLinear
[pattern _ & {:as layers}] ;;ignoring activation 
    (doto pattern  
         (.setInputNeurons  (:input layers))
         (.setOutputNeurons (:output layers)))
    (.generate pattern))

(defmethod network BAMPattern ;no hidden layers - only ActivationBiPolar
[pattern _ & {:as layers}] ;;ignoring layers and activation 
    (doto pattern  
         (.setF1Neurons (:input layers))
         (.setF2Neurons (:output layers)))
    (.generate pattern))
 
(defmethod network BoltzmannPattern ;;no hidden layers - only ActivationBipolar
[pattern _ & {:as layers}] ;;ignoring activation 
    (do  
    (.setInputNeurons pattern  (:input layers))
    (.generate pattern)))
 
(defmethod network CPNPattern ;;one hidden layer - only ActivationBipolar + Competitive
[pattern _ & {:as layers}] ;;ignoring activation 
    (doto pattern  
         (.setInputNeurons  (:input layers))
         (.setInstarCount   ((:hidden layers) 0))
         (.setOutputNeurons (:output layers)))
    (.generate pattern))
 
(defmethod network ElmanPattern;;one hidden layer only - settable activation
[pattern activation & {:as layers}] 
    (doto pattern  
         (.setInputNeurons  (:input layers))
         (.setOutputNeurons (:output layers))
         (.addHiddenLayer ((:hidden layers) 0))
         (.setActivationFunction activation)) 
    (.generate pattern))


(defmethod network HopfieldPattern ;;one hidden layer only - settable activation
[pattern activation & {:as layers}] 
    (doto pattern 
         (.setInputNeurons  (:input layers))
         (.setOutputNeurons (:output layers))
         (.addHiddenLayer   ((:hidden layers) 0))
         (.setActivationFunction activation))
    (.generate pattern))   
 
 
(defmethod network JordanPattern ;;one hidden layer only - settable activation
[pattern activation & {:as layers}] 
    (doto pattern 
         (.setInputNeurons (:input layers))
         (.setOutputNeurons (:output layers))
         (.addHiddenLayer  ((:hidden layers) 0))
         (.setActivationFunction pattern activation))
   (.generate pattern))


(defmethod network SOMPattern ;non settable activation - no hidden layers
[pattern _  & {:as layers}] ;;ignoring activation
    (doto pattern  
         (.setInputNeurons  (:input layers))
         (.setOutputNeurons (:output layers)))
    (.generate pattern))
 
(defmethod network RadialBasisPattern ;usually has one hidden layer - non settable activation
[pattern _  & {:as layers}] ;;ignoring activation
    (doto pattern  
         (.setInputNeurons  (:input layers))
         (.setOutputNeurons (:output layers))
         (.addHiddenLayer ((:hidden layers) 0)))
    (.generate pattern))
    
    
;(defmethod network RSOMPattern ;;no hidden layers - non settable activation
;[layers _ p] ;;ignoring activation
; (let [pattern p]
;    (do  (.setInputNeurons pattern  (:input layers))
;         (.setOutputNeurons pattern (:output layers)) 
;    (. pattern generate))))  
;----------------------------------------------------------------------------------
;----------------------------------------------------------------------------------
;;The next 2 patterns are slightly different than the rest. When called they will return a function (not a pattern object). 
;;This function needs to be called again (with nil arguments for defaults) in order to get the actual pattern object.
    
(defmethod network SVMPattern ;;no hidden layers - non settable activation
[pattern _  & {:as layers}] ;;ignoring activation
 (fn [svm-type kernel-type] ;;returns a function which will return the actual network
    (doto pattern 
         (.setInputNeurons (:input layers))
         (.setOutputNeurons 1) ;only one output is allowed
         (when-not (nil? kernel-type) 
                         (.setKernelType kernel-type))
         (when-not (nil? svm-type) 
                         (.setSVMType svm-type))) 
    (.generate pattern)))
    
(defmethod network PNNPattern ;;no hidden layers - only LinearActivation 
[pattern _  & {:as layers}] ;;ignoring activation
(fn [kernel out-model]
    (doto pattern 
         (.setInputNeurons  (:input layers))
         (.setOutputNeurons (:output layers))
         (when-not (nil? kernel)    
                         (.setKernel kernel))
         (when-not (nil? out-model) 
                         (.setOutmodel out-model))) 
    (.generate pattern)))   
;-----------------------------------------------------------------------------------    

(defmethod network :default 
[_ _ _] ;;ignoring everything
(throw (IllegalArgumentException. "Unsupported pattern!")))
