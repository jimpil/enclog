(ns enclog.examples
(:use [enclog.nnets]
      [enclog.training]
      [enclog.normalization])
(:import ;(org.encog.ml.train.strategy RequiredImprovementStrategy)
         (org.encog.neural.networks.training TrainingSetScore)
         (org.encog.util EngineArray)
         (org.encog.neural.neat.training NEATTraining)
         (org.encog.neural.neat NEATPopulation NEATNetwork)
         (org.encog.ml.kmeans KMeansClustering)
         (org.encog.ml.genetic.genome Genome CalculateGenomeScore)
         ;(org.encog.util.simple EncogUtility)
         (java.text NumberFormat)))
;--------------------------------------*XOR-CLASSIC*------------------------------------------------------------
(defn xor 
"The classic XOR example from the encog book/wiki."
[^Boolean train-to-error?]
(let [xor-input [[0.0 0.0] [1.0 0.0] [0.0 0.1] [1.0 1.0]]
      xor-ideal [[0.0] [1.0] [1.0] [0.0]] 
      dataset   (data :basic-dataset xor-input xor-ideal)
      network   (network  (activation :sigmoid) 
                               (neural-pattern :feed-forward)
                               :input   2
                               :output  1
                               :hidden [2]) ;a single hidden layer
      trainer   ((trainer :resilient-prop) network dataset)]
     ;;make use of the boolean parameter
      (if train-to-error? 
         (train trainer 0.01 []) ;train to max error regardless of iterations
         (train trainer 0.01 300 [] #_[(RequiredImprovementStrategy. 5)])) ;;train to max iterations and max error
     (do (println "\nNeural Network Results:")
     (evaluate network dataset)))) 
      
    ;(loop [t false counter 0 _ nil] 
    ;  (if t (println "Nailed it after" (str counter) "times!")
    ;  (recur  (train trainer 0.001 #_300 [] #_[(RequiredImprovementStrategy. 5)])  ;;train the network until it succeeds
    ;          (inc counter) (. network reset))))     

    
  ;  (doseq [pair dataset] 
  ;  (let [output (. network compute (. pair getInput ))] ;;test the network
  ;       (println (.getData (.getInput pair) 0) "," (. (. pair getInput) getData  1) 
  ;                                               ", actual=" (. output getData  0) 
  ;                                              ", ideal=" (.getData (. pair getIdeal) 0))))     


;---------------------------------------------------------------------------------------------------------
;------------------------------------*XOR-NEAT*-----------------------------------------------------------

(defn xor-neat
"The classic XOR example solved using NeuroEvolution of Augmenting Topologies (NEAT)."
[train-to-error?]
(let [xor-input [[0.0 0.0] [1.0 0.0] [0.0 0.1] [1.0 1.0]]
      xor-ideal [[0.0] [1.0] [1.0] [0.0]] 
      dataset    (data :basic-dataset xor-input xor-ideal)
      activation (activation :step)
      population (NEATPopulation. 2 1 1000)
      trainer    (NEATTraining. (TrainingSetScore. dataset) population)]
                ;((trainer :neat) some-function-name true/false population)
                ;this is the alternative when you have an actual Clojure function that you wan to use
                ;as the fitness-function. Here the implementation is already provided in a Java class,
                ;that is why I'm not using my own function instead.
      (do (.setCenter activation 0.5)
          (.setNeatActivationFunction population activation)) 
      (let [best ^NEATNetwork (if train-to-error? 
                                 (train trainer 0.01 [])       ;;error-tolerance = 1%
                                 (train trainer 0.01 200 []))] ;;iteration limit = 200
              (do (println "\nNeat Network results:")
                  (evaluate best dataset))) ))
      

;----------------------------------------------------------------------------------------------------------
;----------------------------------*LUNAR-LANDER*-----------------------------------------------------------
; this example requires that you have LanderSimulation.class NeuralPilot.class in your classpath.
; both of them are in the jar.

(defmacro pilot-score 
"The fitness function for the GA. You will usually pass your own to the GA. A macro that simply 
wraps a call to your real fitness-function (like here) seems a good choice." 
[network] 
`(. (NeuralPilot. ~network false) scorePilot))

(defn try-it [best-evolved] 
(println"\nHow the winning network landed:")
(let [evolved-pilot (NeuralPilot. best-evolved true)]
(println (. evolved-pilot scorePilot))))


(defn lunar-lander 
"The Lunar-Lander example which can be trained with a GA/simulated annealing. Returns the best evolved network."
[popu]
(let [network (network  (activation :tanh) 
                             (neural-pattern :feed-forward)
                             :input   3
                             :output  1
                             :hidden [50]) ;a single hidden layer of 50 neurons
      trainer   ((trainer :genetic) network (randomizer :nguyen-widrow) 
                                                 (pilot-score network) false  popu 0.1 0.25)
     ]    
     (loop [epoch 1
            _     nil
            best  nil]
     (if (> epoch 200)  (do (.shutdown (org.encog.Encog/getInstance)) best) ;;return the best evolved network 
     (recur (inc epoch) (. trainer iteration) (. trainer getMethod)))) ))
;---------------------------------------------------------------------------------------------------------------
;----------------------------PREDICT-SUNSPOT-SVM------------------------------------------------------------

(def sunspots 
           [0.0262,  0.0575,  0.0837,  0.1203,  0.1883,  0.3033,  
            0.1517,  0.1046,  0.0523,  0.0418,  0.0157,  0.0000,  
            0.0000,  0.0105,  0.0575,  0.1412,  0.2458,  0.3295,  
            0.3138,  0.2040,  0.1464,  0.1360,  0.1151,  0.0575,  
            0.1098,  0.2092,  0.4079,  0.6381,  0.5387,  0.3818,  
            0.2458,  0.1831,  0.0575,  0.0262,  0.0837,  0.1778,  
            0.3661,  0.4236,  0.5805,  0.5282,  0.3818,  0.2092,  
            0.1046,  0.0837,  0.0262,  0.0575,  0.1151,  0.2092,  
            0.3138,  0.4231,  0.4362,  0.2495,  0.2500,  0.1606,  
            0.0638,  0.0502,  0.0534,  0.1700,  0.2489,  0.2824,  
            0.3290,  0.4493,  0.3201,  0.2359,  0.1904,  0.1093,  
            0.0596,  0.1977,  0.3651,  0.5549,  0.5272,  0.4268,  
            0.3478,  0.1820,  0.1600,  0.0366,  0.1036,  0.4838,  
            0.8075,  0.6585,  0.4435,  0.3562,  0.2014,  0.1192,  
            0.0534,  0.1260,  0.4336,  0.6904,  0.6846,  0.6177,  
            0.4702,  0.3483,  0.3138,  0.2453,  0.2144,  0.1114,  
            0.0837,  0.0335,  0.0214,  0.0356,  0.0758,  0.1778,  
            0.2354,  0.2254,  0.2484,  0.2207,  0.1470,  0.0528,  
            0.0424,  0.0131,  0.0000,  0.0073,  0.0262,  0.0638,  
            0.0727,  0.1851,  0.2395,  0.2150,  0.1574,  0.1250,  
            0.0816,  0.0345,  0.0209,  0.0094,  0.0445,  0.0868,  
            0.1898,  0.2594,  0.3358,  0.3504,  0.3708,  0.2500,  
            0.1438,  0.0445,  0.0690,  0.2976,  0.6354,  0.7233,  
            0.5397,  0.4482,  0.3379,  0.1919,  0.1266,  0.0560,  
            0.0785,  0.2097,  0.3216,  0.5152,  0.6522,  0.5036,  
            0.3483,  0.3373,  0.2829,  0.2040,  0.1077,  0.0350,  
            0.0225,  0.1187,  0.2866,  0.4906,  0.5010,  0.4038,  
            0.3091,  0.2301,  0.2458,  0.1595,  0.0853,  0.0382,  
            0.1966,  0.3870,  0.7270,  0.5816,  0.5314,  0.3462,  
            0.2338,  0.0889,  0.0591,  0.0649,  0.0178,  0.0314,  
            0.1689,  0.2840,  0.3122,  0.3332,  0.3321,  0.2730,  
            0.1328,  0.0685,  0.0356,  0.0330,  0.0371,  0.1862,  
            0.3818,  0.4451,  0.4079,  0.3347,  0.2186,  0.1370,  
            0.1396,  0.0633,  0.0497,  0.0141,  0.0262,  0.1276,  
            0.2197,  0.3321,  0.2814,  0.3243,  0.2537,  0.2296,  
            0.0973,  0.0298,  0.0188,  0.0073,  0.0502,  0.2479,  
            0.2986,  0.5434,  0.4215,  0.3326,  0.1966,  0.1365,  
            0.0743,  0.0303,  0.0873,  0.2317,  0.3342,  0.3609,  
            0.4069,  0.3394,  0.1867,  0.1109,  0.0581,  0.0298,  
            0.0455,  0.1888,  0.4168,  0.5983,  0.5732,  0.4644,  
            0.3546,  0.2484,  0.1600,  0.0853,  0.0502,  0.1736,  
            0.4843,  0.7929,  0.7128,  0.7045,  0.4388,  0.3630,  
            0.1647,  0.0727,  0.0230,  0.1987,  0.7411,  0.9947,  
            0.9665,  0.8316,  0.5873,  0.2819,  0.1961,  0.1459,  
            0.0534,  0.0790,  0.2458,  0.4906,  0.5539,  0.5518,  
            0.5465,  0.3483,  0.3603,  0.1987,  0.1804,  0.0811,  
            0.0659,  0.1428,  0.4838,  0.8127]) 
                                  
            
(defn predict-sunspot 
"The PredictSunSpots SVM example ported to Clojure. Not so trivial as the others because it involves temporal data."
[spots]
(let [start-year  1700
      window-size 30 ;input layer count
      train-end 259
      evaluation-end (dec (count spots))
      normalizedSunspots (prepare :array-range nil nil :raw-seq spots :ceiling 0.9 :floor 0.1);using quick method
      closedLoopSunspots (EngineArray/arrayCopy normalizedSunspots)
      train-set         ((data :temporal-window normalizedSunspots) 
                         window-size 1) 
      network ((network (activation :tanh) 
                             (neural-pattern :svm)
                              :input   window-size
                              :output  1      
                              :hidden [0]) nil nil)  ;;passing nil so default values are given for svm/kernel type
      trainer                ((trainer :svm) network train-set) 
      nf                     (NumberFormat/getNumberInstance)]
(do (. trainer iteration) ;;SVM TRAINED AND READY FOR PREDICTIONS AFTER THIS LINE
    (. nf setMaximumFractionDigits 4)
    (. nf setMinimumFractionDigits 4)
    (println "Year" \tab "Actual" \tab "Predict" \tab "Closed Loop Predict")     
(loop [evaluation-start (inc train-end)]          
(if (== evaluation-start evaluation-end) 'DONE...
    (let [input (data :basic window-size)]
    (dotimes [i (. input size)] 
    (. input setData i 
            (aget normalizedSunspots (+ i (- evaluation-start window-size)))))
              (let [output (. network compute input)
                    prediction (. output getData 0)
                    _          (aset closedLoopSunspots evaluation-start prediction)]                   
                    (dotimes [y (. input size)]
                    (. input setData y 
                             (aget closedLoopSunspots  (+ y (- evaluation-start window-size)))))
                              (let [output2 (. network compute input)
                              closed-loop (. output2 getData 0)]
                              (println  (+ start-year evaluation-start)
                                        \tab (. nf format (aget normalizedSunspots evaluation-start))
                                        \tab (. nf format prediction)
                                        \tab (. nf format closed-loop)) #_(debug-repl)) )
(recur (inc evaluation-start))))))))
;--------------------------------------------------------------------------------------------------------------
;--------------------------------*SIMPLE SOM EXAMPLE*----------------------------------------------------------
            
(defn simple-som []
(let [input [[-1.0, -1.0, 1.0, 1.0 ] 
             [1.0, 1.0, -1.0, -1.0]]
      dataset (data :basic-dataset input nil);there is no ideal data (unsupervised)
      network (network nil (neural-pattern :som) :input 4 :output 2)
      trainer ((trainer :basic-som) network 0.7 dataset (make-neighborhoodF :single))      
     ]
     ;(do (train trainer 0.1 10))
     (dotimes [i 10] (.iteration trainer)) ;training complete
     (let [d1 (data :basic (first input))
           d2 (data :basic (second input))]
           (do (println "Pattern 1 winner:" (. network classify d1)) 
               (println "Pattern 2 winner:" (. network classify d2))  
           network)) ;returns the trained network at the end    
))            

;---------------------------------------------------------------------------------------------------------------
;--------------------------------*NORMALIZATION EXAMPLES*--------------------------------------------------------
(def dummy1 [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0])
(def dummy2 [[1.0 2.0 -3.0 4.0 5.0] 
           [ -11.0 12.0 -13.0 14.0 -15.0] 
           [ -2 5 7 -9 8]])

(defn norm-ex-1d [] ;;example with 1d array
(let [source dummy1
      length (count source) 
      raw-input     (input source :forNetwork? false :type :array-1d)         
      input-field   (raw-input);no parameters are necessary here
      output-field  ((output input-field  :type :range-mapped) 0.1 0.9) ;low & high  
      target        (target-storage :norm-array length)
      ready         ((prepare :range [input-field] ;needs to be seqable
                                     [output-field] ;same here 
                                     :ceiling 0.9 
                                     :floor   0.1) false target)]                        
(println   (seq ready) "\n---------- THESE 2 SHOULD BE IDENTICAL! ----------------\n" )

;the version below skips initialising input/output fields and storage targets...
;Uses arrays directly but only supports 1 dimension.
(seq (prepare :array-range nil nil ;inputs & outputs are nil
                :raw-seq source 
                :forNetwork? false
                :ceiling 0.9
                :floor 0.1))
))



(defn norm-ex-2d [] ;;example with 2d array
(let [source dummy2
      column-length (count (first source)) ;5
      row-length  (count source) ;3
      ;ceiling 0.9 
      ;floor 0.1 
      raw-input        (input source :forNetwork? false :type :array-2d)
      input-fields-2d  (for [i (range column-length)] (raw-input i)) ;i=index2         
      ;input-field   (raw-input);
      output-fields-2d   (for [i input-fields-2d] ((output i :type :range-mapped) 0.1 0.9)) ;low-high
      ;output-fields-2d  (raw-output 0.1 0.9)  
      target           (target-storage :norm-array2d row-length column-length)
      ready          ((prepare :range input-fields-2d ;already seqables
                                      output-fields-2d  
                                      :ceiling 0.9 
                                      :floor   0.1) false target)]                 

                  
(println  (map seq (seq ready)) "\n--------------------------------\n" )

))

(defn norm-csv [] ;using EncogAnalyst
(let [source-filename "test2.txt"
      target-filename "generated2.txt"]
 (do ((prepare :csv-range nil nil) source-filename 
                                   target-filename true) 
      (println "DONE!"))))
 
 
(defn norm-csv2 []  ;using input/output fields and storage
(let [source "test2.txt"
      target "generated-MY.txt"
      columns 4
      max (- 0.1)
      min (- 0.9)
      input-fields (for [i (range columns)] 
                      ((input (java.io.File. source) :forNetwork? false 
                                      :type :csv 
                                      :column-offset i)))
      output-fields (for [i input-fields] ((output i :type :range-mapped) min max))
      storage ((target-storage :norm-csv) target)
      ready ((prepare :range input-fields 
                             output-fields
                             :ceiling max
                             :floor   min) false storage)] 
    ))


;------------------------------------------------------------------------------------------------------------
;-----------------------------*CLUSTERING EXAMPLE*-----------------------------------------------------------

(defn kmeans [k]  ;simple k-means clustering 
(let [raw-data [[28 15 22] [16 15 32] [32 20 44] [1 2 3] [3 2 1]] ;the data to be clustered
      wrapped-data (map #(data :basic %) raw-data)   ;wrap each inner vector into a BasicMLData object
      dataset (let [ds (data :basic-dataset)] (doseq [el wrapped-data] (.add ds el)) ds) ;make dataset with no ideal data
      kmeans  (KMeansClustering. k dataset) ;the actual K-Means object
      ready   (.iteration kmeans 100) ;100 iterations is more than enough for this toy-dataset
      clusters (.getClusters kmeans) ; the actual clusters - an array of MLCluster objects
      numClusters (.numClusters kmeans)] ; the total number of clusters (k)
      {:number-of-clusters numClusters 
       :clusters clusters 
       :cluster1-contents (str (.getData (first clusters)))
       :cluster2-contents (str (.getData (second clusters)))}
       ))


;---------------------------------------------------------------------------------------------------------------
;--------------------------------TRAVELLING-SALESMAN-PROBLEM*---------------------------------------------------

(defprotocol Proximable
  "Satisfiers can calculate their proximity to others."
  (proximity [this ox oy] [this ocity] 
  "Calculates proximity to another city either by passing the coordinates or the other city itself."))

(defrecord City [x y] ;represent each city as a record with 2 coordinates
Proximable
(proximity [this ox oy] (let [xdiff (- (:x this) ox)
                              ydiff (- (:y this) oy)] 
(int (Math/sqrt (+ (Math/pow xdiff 2) 
                   (Math/pow ydiff 2))))))
(proximity [this ocity] (proximity this (:x ocity) (:y ocity))))

(defn place-cities 
"Initialise cities in random locations." [n]
(let [map-size 256]
(vec (for [_ (range n)] 
     (City. (int (* (rand) map-size))      ;x random coord
            (int (* (rand) map-size))))))) ;y random coord

(def cities (place-cities 70))

(def tsp-fitness
(reify CalculateGenomeScore 
  (^double calculateScore [this ^Genome genome] 
     (let [path (.getOrganism genome)]
       (loop [i 0
              dests cities
              city1 (dests i)
              city2 (dests (inc i))
              total-distance 0]
       (if-not (seq dests) total-distance ;return total distance if run out of destinations
       (recur (inc i)
              (rest dests) 
              (dests (get path (inc i)))
              (dests (get path (+ i 2))) 
              (+ total-distance (proximity city1 city2)))))))
  (^boolean shouldMinimize [this] true)))

#_(defn solve-tsp "Solve the tsp problem with a genetic-algorithm." [pop-size]
(let [ga (doto (BasicGeneticAlgorithm.) 
               (.setMutationPercent 0.1)
               (.setPercentToMate  0.25)
               (.setMatingPopulation 0.5)
               (.setCrossover (SpliceNoRepeat. ((comp #(/ % 3) count) cities)))
               (.setMutate (MutateShuffle.))
               (.setCalculateScore tsp-fitness)
               (.setPopulation (BasicPopulation. pop-size)))]
(dotimes [_ pop-size]
(let [genome (TSPGenome. ga (into-array cities))]
  (.getPopulation ga (.add genome))
  (.calculateScore ga genome)))
    (do (.claim population ga) 
        (.sort population))))



























;---------------------------------------------------------------------------------------------------
;run the lunar lander example using main otherwise the repl will hang under leiningen. 
(defn -main [] 
;(xor true)
(xor false)
(xor-neat false)
(kmeans 2)
(predict-sunspot sunspots)
(try-it (lunar-lander 1000))
)
