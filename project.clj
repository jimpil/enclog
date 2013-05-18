(defproject enclog "0.5.8"
  :description "Thin Clojure wrapper for Encog(v3) Machine-Learning framework."
  :url "http://github.com/jimpil/enclog"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [org.encog/encog-core "3.1.0"]
                 ]
  :jvm-opts ["-Xmx1g"] 
  ;:javac-options {:classpath "target/dependency/encog-core-3.1.0.jar" :destdir "target/classes"}
  ;:java-source-paths ["src/encog_java"]
  ;:main     enclog.examples
  )
