package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

type Pairs struct {
	vectorA []int64
	vectorB []int64
}

func fillVector(vectorDim int, universe int, r *rand.Rand) []int64 {
	vector := make([]int64, vectorDim)

	for i := 0; i < vectorDim; i++ {
		vector[i] = int64(r.Intn(universe))
	}

	return vector
}

func fillPairs(vectorDim int, universe int, seed int64) Pairs {
	var a, b []int64
	var wg sync.WaitGroup

	wg.Add(2)
	go func() {
		defer wg.Done()
		r := rand.New(rand.NewSource(seed))
		a = fillVector(vectorDim, universe, r)
	}()
	go func() {
		defer wg.Done()
		r := rand.New(rand.NewSource(seed + 100))
		b = fillVector(vectorDim, universe, r)
	}()
	wg.Wait()

	return Pairs{vectorA: a, vectorB: b}
}

func cosineSeq(pair Pairs) float64 {
	var dotProduct, sum2A, sum2B float64

	for i := 0; i < len(pair.vectorA); i++ {
		va := float64(pair.vectorA[i])
		vb := float64(pair.vectorB[i])

		dotProduct += va * vb
		sum2A += va * va
		sum2B += vb * vb
	}

	return dotProduct / (math.Sqrt(sum2A) * math.Sqrt(sum2B))
}

func cosineCon(pair Pairs) float64 {
	return 1
}

func pearsonSeq(pair Pairs) float64 {
	var dotProduct, sumA, sumB, sum2A, sum2B float64
	lenVector := float64(len(pair.vectorA))

	for i := 0; i < len(pair.vectorA); i++ {
		va := float64(pair.vectorA[i])
		vb := float64(pair.vectorB[i])

		dotProduct += va * vb
		sumA += va
		sumB += vb
		sum2A += va * va
		sum2B += vb * vb

	}

	return (lenVector*dotProduct - sumA*sumB) / math.Sqrt((lenVector*sum2A-sumA*sumA)*(lenVector*sum2B-sumB*sumB))
}

func pearsonCon(pair Pairs) float64 {
	return 1
}

func jaccardSeq(pair Pairs) float64 {
	union := make(map[int64]struct{})
	intersection := make(map[int64]struct{})

	setA := make(map[int64]struct{})
	setB := make(map[int64]struct{})

	for _, num := range pair.vectorA {
		setA[num] = struct{}{}
	}

	for _, num := range pair.vectorB {
		setB[num] = struct{}{}
	}

	for number := range setA {
		union[number] = struct{}{}
		if _, found := setB[number]; found {
			intersection[number] = struct{}{}
		}
	}

	for number := range setB {
		union[number] = struct{}{}
	}

	if len(union) == 0 {
		return 0.0
	}
	return float64(len(intersection)) / float64(len(union))
}

func jaccardCon(pair Pairs) float64 {
	return 1
}

func main() {
	seed := flag.Int64("seed", 42, "Random Seed")
	vectorDim := flag.Int("dim", 9999999, "Array Dimension, x")
	algorithm := flag.String("algorithm", "cosine", "Select Algorithm: cosine | pearson | jaccard")

	flag.Parse()
	var universe int = 10000

	pairs := fillPairs(*vectorDim, universe, *seed)

	switch *algorithm {
	case "cosine":
		t0 := time.Now()
		cseq := float64(cosineSeq(pairs))

		fmt.Printf("Cosine Seq %f \n", cseq)
		tSeq := time.Since(t0)
		fmt.Printf("Time duration %v ms \n", tSeq)

	case "pearson":
		t0 := time.Now()
		cseq := float64(pearsonSeq(pairs))

		fmt.Printf("Pearson Seq %f \n", cseq)
		tSeq := time.Since(t0)
		fmt.Printf("Time duration %v ms \n", tSeq)

	case "jaccard":
		t0 := time.Now()
		cseq := float64(jaccardSeq(pairs))

		fmt.Printf("Jaccard Seq %f \n", cseq)
		tSeq := time.Since(t0)
		fmt.Printf("Time duration %v ms \n", tSeq)

	default:
		print("Tipo de algoritmo incorrecto")
	}
}
