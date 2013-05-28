(defproject enclog "0.6.2"
  :description "Thin Clojure wrapper for Encog(v3) Machine-Learning framework."
  :url "http://github.com/jimpil/enclog"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [org.encog/encog-core "3.1.0"]]
  :jvm-opts ["-Xmx1g"]
  :scm  {:name "git"
         :url "https://github.com/jimpil/enclog"} 
  ;:java-source-paths ["src/encog_java"]
  ;:main     enclog.examples
  )
