package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

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

func main() {
	vecA := makeRandomVector(1000000, 3.0)
	vecB := makeRandomVector(1000000, 4.0)
	jaccardA := make([]float64, 1000000)
	jaccardB := make([]float64, 1000000)
	for i := range jaccardA {
		jaccardA[i] = float64(rand.Intn(2))
		jaccardB[i] = float64(rand.Intn(2))
	}

	var wg sync.WaitGroup

	start := time.Now()
	jaccardIndexVal := jaccardIndex(jaccardA, jaccardB)
	similarity := cosineSimilarity(vecA, vecB)
	pearson := pearsonCorrelation(vecA, vecB)

	fmt.Printf("Cosine Similarity: %f\n", similarity)
	fmt.Printf("Pearson Correlation: %f\n", pearson)
	fmt.Printf("Jaccard Index: %f\n", jaccardIndexVal)

	elapsed := time.Since(start)
	fmt.Printf("Tiempo transcurrido (secuencial): %f\n", elapsed.Seconds())

	startC := time.Now()

	wg.Add(3)

	go func() {
		defer wg.Done()
		jaccardIndexVal = jaccardIndex(jaccardA, jaccardB)
	}()
	go func() {
		defer wg.Done()
		similarity = cosineSimilarity(vecA, vecB)
	}()
	go func() {
		defer wg.Done()
		pearson = pearsonCorrelation(vecA, vecB)
	}()

	wg.Wait()

	elapsedC := time.Since(startC)

	fmt.Printf("Tiempo transcurrido (concurrente): %f\n", elapsedC.Seconds())

	fmt.Printf("Speedup: %f\n", elapsed.Seconds()/elapsedC.Seconds())

}
