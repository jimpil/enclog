(defproject enclog "0.6.6"
  :description "Thin Clojure wrapper for Encog(v3) Machine-Learning framework."
  :url "http://github.com/jimpil/enclog"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.encog/encog-core "3.3.0"]]
  :jvm-opts ["-Xmx1g"]
  :scm  {:name "git"
         :url "https://github.com/jimpil/enclog"}

  :source-paths ["src" "src/enclog"]
  :java-source-paths ["test/enclog/"]
  :test-paths ["test" "test/enclog"]
  )
