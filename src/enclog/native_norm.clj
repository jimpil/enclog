(ns enclog.native_norm
   (:require [clojure.pprint :refer [pprint]] 
             [clojure.core.reducers :as r]
             [clojure.test :refer [with-test is testing run-tests]]) )
;-----------------------------------------------------------------------------------------------------------------------------------             
;----------------------------<EXPERIMENTAL CODE>------------------------------------------------------------------------------------             

(set! *warn-on-reflection* true)
(set! *unchecked-math* true)

(defprotocol Normalisable
(normalise [this transform-fn] 
           [this transform-fn limits]))
           
(defn transform-val "A generic tranformation fn. Expects the formula to use (another fn), an element and potential extra arguments."
([formula x & opts]  (apply formula x opts))
([formula x] (transform-val formula x  1 -1)) )          

;;High-performance extension points for all major Clojure data-structures including arrays [ints, floats, longs & doubles]
;;whatever collection type you pass in, the same type you will get back unless nothing covers it, in which case a lazy-seq will be most likely returned.
;;If you pass a non-persistent java.util.List object, you'll get a persistent vector back.
(extend-protocol Normalisable
      
Number
(normalise 
 ([this transform] (normalise this #(apply transform %1 %2) [[1 -1] [10 -10]]))
 ([this transform limits] (transform this limits)))
 
String
(normalise 
([this stemmer] 
  (normalise this stemmer "english"))
([this stemmer lang] 
  (stemmer this lang)))
  
java.util.List ;;if this fires, we're dealing with a Java list-like collection - return a vector
(normalise
([this transform]
   (let [top    (delay (apply max this))
         bottom (delay (apply min this))]
   (mapv #(normalise % transform [top bottom]) this)) )
([this transform limits]
   (normalise this #(transform %1 limits %2))) )  
  
clojure.lang.IPersistentCollection;;if this fires, we don't know the type so  we'll return a lazy-seq
(normalise
([this transform]
   (let [top    (delay (apply max this))
         bottom (delay (apply min this))]
   (map #(normalise % transform [top bottom]) this)) )
([this transform limits]
   (normalise this #(transform %1 limits %2))) )
     
clojure.lang.PersistentList
(normalise
([this transform]
   (let [top    (delay (apply max this))
         bottom (delay (apply min this))]
    (->> (mapv #(normalise % transform [top bottom]) this)
       rseq
      (into '()))) )
([this transform limits]
   (normalise this #(transform %1 limits %2))) ) 
    
clojure.lang.LazySeq
(normalise
([this transform]
   (let [top    (delay (apply max this))
         bottom (delay (apply min this))]
   (map #(normalise % transform [top bottom]) this)) )
([this transform limits]
   (normalise this #(transform %1 limits %2))) )
      
clojure.lang.IPersistentVector
(normalise
([this transform]
 (let [top    (delay (apply max this))
       bottom (delay (apply min this))]
 (if (> 1000 (count this)) 
   (mapv #(normalise % transform [top bottom]) this);;do it serially in one pass           
   (into []                                         ;;do it in parallel using reducers
     (r/foldcat (r/map #(normalise % transform [top bottom]) this))))) )
([this transform limits]
   (normalise this #(transform %1 limits %2))) ) 
     
clojure.lang.IPersistentSet ;;sets are typically not ordered so ordering will dissapear after processing
(normalise
([this transform]
   (let [top    (delay (apply max this))
         bottom (delay (apply min this))]
 (persistent!        
   (reduce #(conj! %1 (normalise %2 transform [top bottom])) (transient #{}) this))))
([this transform limits]
   (normalise this #(transform %1 limits %2))) )
   
clojure.lang.IPersistentMap ;;assuming a map with collections for keys AND values (a dataset perhaps?)
(normalise
([this transform]
 (persistent!        
   (reduce-kv #(assoc! %1 (normalise %2 transform) 
                          (normalise %3 transform)) (transient {}) this)))
([this transform limits]
   (normalise this #(transform %1 limits %2))))  )
   
(extend-protocol Normalisable   
(Class/forName "[D")  
(normalise
([this transform]
   (let [top    (delay (apply max this))
         bottom (delay (apply min this))]
   (amap ^doubles this idx ret (double (normalise (aget ^doubles this idx) transform [top bottom])))))
([this transform limits]
   (normalise this #(transform %1 limits %2)))) )  
   
(extend-protocol Normalisable   
(Class/forName "[F")  
(normalise
([this transform]
   (let [top    (delay (apply max this))
         bottom (delay (apply min this))]
   (amap ^floats this idx ret (float (normalise (aget ^floats this idx) transform [top bottom])))))
([this transform limits]
   (normalise this #(transform %1 limits %2)))) )
   
(extend-protocol Normalisable    
(Class/forName "[J")
(normalise
([this transform]
   (let [top    (delay (apply max this))
         bottom (delay (apply min this))]
   (amap ^longs this idx ret (long (normalise (aget ^longs this idx) transform [top bottom])))))
([this transform limits]
   (normalise this #(transform %1 limits %2)))) )
      
(extend-protocol Normalisable
(Class/forName "[I")
(normalise
([this transform]
   (let [top    (delay (apply max this))
         bottom (delay (apply min this))]
   (amap ^ints this idx ret (int (normalise (aget ^ints this idx) transform [top bottom])))))
([this transform limits]
   (normalise this #(transform %1 limits %2)))) )
   
;;this is how client code would look like   
(defn in-range-formula 
"The most common normalisation technique (for numbers)." 
([x [tn bn] [ti bi]] 
(+ bn
   (* (/ (- x (force bi)) 
         (- (force ti) (force bi))) 
      (- tn bn))) )
([x [ti bi]] 
  (in-range-formula x [1 -1] [ti bi]))
([x] 
  (in-range-formula x [1 -1] [10 -10])) )

(def transform-in-range "In-range transformer." (partial transform-val in-range-formula))

(defn reciprocal-formula 
"Reciprocal normalization is always normalizing to a number in the range between 0 and 1.
 It should only be used to normalize numbers greater than 1. Do NOT pass in 0!"
([x _ _]
  (/ 1 x))
([x _]
(reciprocal-formula x _ nil))
([x] 
 (reciprocal-formula x nil)) )
 
(def transform-reciprocal "Reciprocal transformer." (partial transform-val reciprocal-formula)) 
 
(defn divide-by-value-formula
"Nrmalises by dividing all elements by the given value." 
([div-val x _ ]
  (/ x div-val))
([div-val x]
  (divide-by-value-formula div-val x nil)) )       

(def transform-by-value "Divide-by-value transformer."  (partial transform-val (partial divide-by-value-formula 100))) 
   
