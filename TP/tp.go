package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

type datasetRow struct {
	index                          int64
	app_id                         int64
	app_name                       string
	review_id                      int64
	language                       string
	review                         string
	timestamp_created              int64
	timestamp_updated              int64
	recommended                    bool
	votes_helpful                  int64
	votes_funny                    int64
	weighted_vote_score            float64
	comment_count                  int64
	steam_purchase                 bool
	received_for_free              bool
	written_during_early_access    bool
	author_steamid                 int64
	author_num_games_owned         int64
	author_num_reviews             int64
	author_playtime_forever        float64
	author_playtime_last_two_weeks float64
	author_playtime_at_review      float64
	author_last_played             float64
}

func parseInt64(s string) int64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0
	}
	return v
}

func parseFloat64(s string) float64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return v
}

func parseBool(s string) bool {
	s = strings.TrimSpace(s)
	// accept true/false (any case) and 1/0
	if s == "1" {
		return true
	}
	if s == "0" {
		return false
	}
	return strings.EqualFold(s, "true")
}

func main() {
	dir, err := os.Getwd()
	fmt.Println(dir)

	file, err := os.Open("data/steam_reviews.csv")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()

	var dataset []datasetRow

	reader := csv.NewReader(file)
	for i := 0; i < 100; i++ {
		if i == 0 {
			record, _ := reader.Read()
			print(record)
		}
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Println("Error reading record:", err)
			return
		}

		fmt.Println(record[0])

		row := datasetRow{
			index:                          parseInt64(record[0]),
			app_id:                         parseInt64(record[1]),
			app_name:                       record[2],
			review_id:                      parseInt64(record[3]),
			language:                       record[4],
			review:                         record[5],
			timestamp_created:              parseInt64(record[6]),
			timestamp_updated:              parseInt64(record[7]),
			recommended:                    parseBool(record[8]),
			votes_helpful:                  parseInt64(record[9]),
			votes_funny:                    parseInt64(record[10]),
			weighted_vote_score:            parseFloat64(record[11]),
			comment_count:                  parseInt64(record[12]),
			steam_purchase:                 parseBool(record[13]),
			received_for_free:              parseBool(record[14]),
			written_during_early_access:    parseBool(record[15]),
			author_steamid:                 parseInt64(record[16]),
			author_num_games_owned:         parseInt64(record[17]),
			author_num_reviews:             parseInt64(record[18]),
			author_playtime_forever:        parseFloat64(record[19]),
			author_playtime_last_two_weeks: parseFloat64(record[20]),
			author_playtime_at_review:      parseFloat64(record[21]),
			author_last_played:             parseFloat64(record[22]),
		}

		dataset = append(dataset, row)
	}

	fmt.Println(dataset[:6])
}
