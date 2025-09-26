package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"
)

//Estructura que define el trabajo -> input
type Job struct {
	id int
	a []float64
	b []float64
}

type Result struct{
	id int
	score float64
	metric string
}

func throwPanic(lenVecA, lenVecB int) {
	if lenVecA != lenVecB {
		panic("Los vectores deben tener la misma cantidad de valores")
	}
}

func jaccardIndex(vecA, vecB []float64) float64 {
	throwPanic(len(vecA), len(vecB))
	var inter, union int
	for i := range vecA {
		if vecA[i] != 0 || vecB[i] != 0 {
			union++
		}
		if vecA[i] != 0 && vecB[i] != 0 {
			inter++
		}
	}
	if union != 0 {
		return float64(inter) / float64(union)
	}

	return 0
}

func pearsonCorrelation(vecA, vecB []float64) float64 {
	var meanA float64 = 0
	var meanB float64 = 0
	var upper float64 = 0
	var cumX float64 = 0
	var cumY float64 = 0

	throwPanic(len(vecA), len(vecB))

	for i := range vecA {
		meanA = meanA + vecA[i]
		meanB = meanB + vecB[i]
	}

	meanA = meanA / float64(len(vecA))
	meanB = meanB / float64(len(vecB))

	for i := range vecA {

		upper = upper + (vecA[i]-meanA)*(vecB[i]-meanB)
		cumX = cumX + math.Pow(vecA[i]-meanA, 2)
		cumY = cumY + math.Pow(vecB[i]-meanB, 2)
	}

	return upper / (math.Sqrt(cumX) * math.Sqrt(cumY))
}

func cosineSimilarity(vecA, vecB []float64) float64 {
	var dotProduct float64 = 0
	var magA float64 = 0
	var magB float64 = 0

	throwPanic(len(vecA), len(vecB))

	for i := range vecA {
		dotProduct = dotProduct + vecA[i]*vecB[i]
		magA = magA + vecA[i]*vecA[i]
		magB = magB + vecB[i]*vecB[i]
	}

	if magA != 0 && magB != 0 {
		return dotProduct / (math.Sqrt(magA) * math.Sqrt(magB))
	}

	return 0
}

func makeRandomVector(n int, max float64) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = rand.Float64() * max
	}
	return v
}

func worker(id int, tasks <- chan Job, results chan <- Result){
	for job := range tasks{
		scoreC := cosineSimilarity(job.a, job.b)
		scoreP := pearsonCorrelation(job.a, job.b)
		scoreJ := jaccardIndex(job.a, job.b)
		results <- Result{id: job.id, score: scoreC, metric : "Cosine"}
		results <- Result{id: job.id, score: scoreP, metric : "Pearson"}
		results <- Result{id: job.id, score: scoreJ, metric : "Jaccard"}

	}
}


func main() {
	rand.Seed(time.Now().UnixNano())
	//Usamos todos los cores de la CPU
	numWorkers := runtime.NumCPU()
	numTasks := 6

	vecA := makeRandomVector(1000000, 3.0)
	vecB := makeRandomVector(1000000, 4.0)
	jaccardA := make([]float64, 1000000)
	jaccardB := make([]float64, 1000000)

	for i := range jaccardA {
		jaccardA[i] = float64(rand.Intn(2))
		jaccardB[i] = float64(rand.Intn(2))
	}

	startC := time.Now()

	tasks := make(chan Job, numWorkers)
	results := make(chan Result, numTasks * 3)

	for i := 0; i < numWorkers; i++ {
	    go worker(i, tasks, results)
	}	
	
	for j := 0; j < numTasks; j++ {
		tasks <- Job{id:j, a:vecA, b:vecB}
	}	
		
	close(tasks)

    for k := 1; k <= numTasks * 3; k++ {
        result := <-results
	fmt.Printf("Result: job=%d, score=%f, metric=%s\n", result.id, result.score, result.metric)
    }

	elapsedC := time.Since(startC)
	fmt.Printf("Tiempo transcurrido (concurrente): %f\n", elapsedC.Seconds())



	start := time.Now()
	jaccardIndexVal := jaccardIndex(jaccardA, jaccardB)
	similarity := cosineSimilarity(vecA, vecB)
	pearson := pearsonCorrelation(vecA, vecB)

	fmt.Printf("Cosine Similarity: %f\n", similarity)
	fmt.Printf("Pearson Correlation: %f\n", pearson)
	fmt.Printf("Jaccard Index: %f\n", jaccardIndexVal)

	elapsed := time.Since(start)
	fmt.Printf("Tiempo transcurrido (secuencial): %f\n", elapsed.Seconds())




	fmt.Printf("Speedup: %f\n", elapsed.Seconds()/elapsedC.Seconds())

}
