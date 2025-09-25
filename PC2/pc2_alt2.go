package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

var amountGoroutines = 150

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

func sumArray(numArray []int64) float64 {
	total := int64(0)

	for _, num := range numArray {
		total += num
	}

	return float64(total)
}

func sumSquareArray(numArray []int64) float64 {
	total := int64(0)

	for _, num := range numArray {
		total += num * num
	}

	return float64(total)
}

func sumDotProduct(numArrayA []int64, numArrayB []int64) float64 {
	total := int64(0)

	for i := 0; i < len(numArrayA); i++ {
		total += numArrayA[i] * numArrayB[i]
	}

	return float64(total)
}

func sumArrayConcurrent(pairArray []int64) float64 {
	num_chunks := amountGoroutines
	sum := make([]int64, num_chunks)

	var wg sync.WaitGroup
	wg.Add(amountGoroutines)

	chunkSize := (len(pairArray) + num_chunks - 1) / num_chunks

	for i := 0; i < num_chunks; i++ {
		go func(i int) {
			defer wg.Done()
			start := i * chunkSize
			end := (i + 1) * chunkSize

			if end > len(pairArray) {
				end = len(pairArray)
			}

			sum[i] = int64(sumArray(pairArray[start:end]))
		}(i)
	}

	wg.Wait()

	return sumArray(sum)
}

func sumSquareArrayConcurrent(pairArray []int64) float64 {
	num_chunks := amountGoroutines
	sum := make([]int64, num_chunks)

	var wg sync.WaitGroup
	wg.Add(amountGoroutines)

	chunkSize := (len(pairArray) + num_chunks - 1) / num_chunks

	for i := 0; i < num_chunks; i++ {
		go func(i int) {
			defer wg.Done()
			start := i * chunkSize
			end := (i + 1) * chunkSize

			if end > len(pairArray) {
				end = len(pairArray)
			}

			sum[i] = int64(sumSquareArray(pairArray[start:end]))
		}(i)
	}

	wg.Wait()

	return sumArray(sum)
}

func sumDotProductConcurrent(pairArrayA []int64, pairArrayB []int64) float64 {
	num_chunks := amountGoroutines
	sum := make([]int64, num_chunks)

	var wg sync.WaitGroup
	wg.Add(amountGoroutines)

	chunkSize := (len(pairArrayA) + num_chunks - 1) / num_chunks

	for i := 0; i < num_chunks; i++ {
		go func(i int) {
			defer wg.Done()
			start := i * chunkSize
			end := (i + 1) * chunkSize

			if end > len(pairArrayA) {
				end = len(pairArrayA)
			}

			sum[i] = int64(sumDotProduct(pairArrayA[start:end], pairArrayB[start:end]))
		}(i)
	}

	wg.Wait()

	return sumArray(sum)
}

func cosineSeq(pair Pairs) float64 {
	dotProduct := sumDotProduct(pair.vectorA, pair.vectorB)
	nA := math.Sqrt(sumSquareArray(pair.vectorA))
	nB := math.Sqrt(sumSquareArray(pair.vectorB))

	return dotProduct / (nA * nB)
}

func cosineCon(pair Pairs) float64 {
	dotProduct := sumDotProductConcurrent(pair.vectorA, pair.vectorB)
	nA := math.Sqrt(sumSquareArrayConcurrent(pair.vectorA))
	nB := math.Sqrt(sumSquareArrayConcurrent(pair.vectorB))

	return dotProduct / (nA * nB)
}

func pearsonSeq(pair Pairs) float64 {
	dotProduct := sumDotProduct(pair.vectorA, pair.vectorB)
	sumA := sumArray(pair.vectorA)
	sumB := sumArray(pair.vectorB)
	sum2A := sumSquareArray(pair.vectorA)
	sum2B := sumSquareArray(pair.vectorB)

	lenVector := float64(len(pair.vectorA))

	return (lenVector*dotProduct - sumA*sumB) / math.Sqrt((lenVector*sum2A-sumA*sumA)*(lenVector*sum2B-sumB*sumB))
}

func pearsonCon(pair Pairs) float64 {
	dotProduct := sumDotProductConcurrent(pair.vectorA, pair.vectorB)
	sumA := sumArrayConcurrent(pair.vectorA)
	sumB := sumArrayConcurrent(pair.vectorB)
	sum2A := sumSquareArrayConcurrent(pair.vectorA)
	sum2B := sumSquareArrayConcurrent(pair.vectorB)

	lenVector := float64(len(pair.vectorA))

	return (lenVector*dotProduct - sumA*sumB) / math.Sqrt((lenVector*sum2A-sumA*sumA)*(lenVector*sum2B-sumB*sumB))
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
	n := len(pair.vectorA)
	numChunks := amountGoroutines

	chunkSize := (n + numChunks - 1) / numChunks

	type counts struct{ inter, cardA, cardB int }
	partial := make([]counts, numChunks)

	var wg sync.WaitGroup
	wg.Add(numChunks)

	for i := 0; i < numChunks; i++ {
		go func(i int) {
			defer wg.Done()
			start := i * chunkSize
			end := (i + 1) * chunkSize
			if end > n {
				end = n
			}

			setA := make(map[int64]struct{})
			setB := make(map[int64]struct{})

			for _, x := range pair.vectorA[start:end] {
				setA[x] = struct{}{}
			}
			for _, y := range pair.vectorA[start:end] {
				setA[y] = struct{}{}
			}

			inter := 0
			for x := range setA {
				if _, ok := setB[x]; ok {
					inter++
				}
			}
			partial[i] = counts{inter: inter, cardA: len(setA), cardB: len(setB)}
		}(i)
	}
	wg.Wait()

	inter, cardA, cardB := 0, 0, 0
	for _, c := range partial {
		inter += c.inter
		cardA += c.cardA
		cardB += c.cardB
	}

	union := cardA + cardB - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)

}

func main() {
	seed := flag.Int64("seed", 42, "Set RNG seed")
	goroutines := flag.Int("goroutines", 100, "Amount of goroutines")
	vectorDim := flag.Int("dim", 999_999_999, "Array Dimension, x")
	algorithm := flag.String("algorithm", "cosine", "Select Algorithm: cosine | pearson | jaccard")

	flag.Parse()

	amountGoroutines = *goroutines

	var universe int = 10000

	pairs := fillPairs(*vectorDim, universe, *seed)

	switch *algorithm {
	case "cosine":
		t0 := time.Now()
		cseq := float64(cosineSeq(pairs))

		fmt.Printf("Cosine Seq %f \n", cseq)
		tSeq := time.Since(t0)
		fmt.Printf("Time duration %v \n", tSeq)

		t0 = time.Now()
		cseq = float64(cosineCon(pairs))

		fmt.Printf("Cosine Con %f \n", cseq)
		tSeq = time.Since(t0)
		fmt.Printf("Time duration %v \n", tSeq)

	case "pearson":
		t0 := time.Now()
		cseq := float64(pearsonSeq(pairs))

		fmt.Printf("Pearson Seq %f \n", cseq)
		tSeq := time.Since(t0)
		fmt.Printf("Time duration %v \n", tSeq)

		t0 = time.Now()
		cseq = float64(pearsonCon(pairs))

		fmt.Printf("Pearson Con %f \n", cseq)
		tSeq = time.Since(t0)
		fmt.Printf("Time duration %v \n", tSeq)
	case "jaccard":
		t0 := time.Now()
		cseq := float64(jaccardSeq(pairs))

		fmt.Printf("Jaccard Seq %f \n", cseq)
		tSeq := time.Since(t0)
		fmt.Printf("Time duration %v \n", tSeq)

		t0 = time.Now()
		cseq = float64(pearsonCon(pairs))

		fmt.Printf("Jaccard Con %f \n", cseq)
		tSeq = time.Since(t0)
		fmt.Printf("Time duration %v \n", tSeq)

	default:
		print("Tipo de algoritmo incorrecto")
	}
}
