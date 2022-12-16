#include "loop_detector.h"

#include <cmath>
#include <fstream>


// LoopDetector::LoopDetector(std::vector<PoseGraphEdge>& edges): m_edges(edges)
// {
//     unsigned int b = 9; // branching factor
//     unsigned int d = 3; // depth levels
//     DBoW2::WeightingType weighting = DBoW2::TF_IDF;
//     DBoW2::ScoringType scoring = DBoW2::L1_NORM;
//
//     m_vocabulary = OrbVocabulary(b, d, weighting, scoring);
//     m_database = OrbDatabase(m_vocabulary, false, 0);
// }

LoopDetector::LoopDetector(
    PoseGraphOptimizer& optimizer,
    std::vector<Frame*>& cam_frames,
    std::vector<PoseGraphEdge>& edges):
    m_optimizer(optimizer),
    m_edges(edges),
    m_cam_frames(cam_frames)
{
    // ensure loop can be detected
    assert(MAX_LOOP_PROBABILITY > THRES_IS_LOOP);

    // ensure loop can be denied
    assert(MIN_LOOP_PROBABILITY < THRES_NOT_LOOP);

    // first frame has no match score
    m_scores.emplace_back(0);
}

LoopDetector::~LoopDetector()
{
    std::ofstream output;
    output.open("loop_matches.txt");

    if (!output.is_open()) return;

    for (int i = 0; i < m_matches.size(); i++)
    {
        output << "frame " << i << "matches with frame " << m_matches[i].first
               << " at score " << m_matches[i].second
               << "probability = " << m_probabilities[i] << std::endl;
    }
}

void LoopDetector::Track(const std::vector<cv::Mat>& features)
{
    m_database.add(features);
}

void LoopDetector::Query(const std::vector<cv::Mat>& features)
{
    unsigned int curr_frame_id = m_cam_frames.size() - 1;

    DBoW2::QueryResults results;
    m_database.query(features, results, QUERY_SIZE_FROM_BOW_DATABASE);

    for (auto result : results)
    {
        if (curr_frame_id - result.Id < THRES_DISTANCE_FRAME) continue;
        float s = ScoreProbability(result.Score);
        float p = MatchProbability(result.Id, result.Score);

        // add potential pose graph constraint
        if (p > THRES_MATCH_PROBABILITY)
        {
            std::cout << "[INFO]: Loop edge detected." << std::endl;
            m_edges.emplace_back(std::make_pair(result.Id, curr_frame_id));
        }
    }
}

void LoopDetector::LoadVocabulary(const std::string& file_path)
{
    m_vocabulary = OrbVocabulary(file_path);
    m_database = OrbDatabase(m_vocabulary, false, 0);
}

void LoopDetector::SaveVocabulary(const std::string& file_path) const
{
    m_vocabulary.save(file_path);
}

float LoopDetector::ScoreProbability(unsigned int score) const
{
    return 0.5;
}

float LoopDetector::MatchProbability(unsigned int id, float score) const
{
    float n_inv = 1 / (float)SCORE_WINDOW;

    float score_s  = 0.0;
    float score_ss = 0.0;

    for (int i = id; i < id + SCORE_WINDOW; i++)
    {
        score_s  += m_scores[i];
        score_ss += m_scores[i] * m_scores[i];
    }

    float score_avg = score_s  * n_inv;
    float score_var = score_ss * n_inv - score_avg * score_avg;

    float score_dev = (score - score_avg) / std::sqrt(score_var);
    float probability = std::erf(score_dev);

    if (probability > 0.5)
    {
        std::cout << probability << std::endl;
    }

    return probability;
}