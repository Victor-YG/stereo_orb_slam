#pragma once

#include "params.h"
#include "pose_graph.h"
#include "camera_frame.h"
#include "pose_graph_optimizer.h"

#include "DBoW2/DBoW2.h"


class LoopDetector
{
public:
    LoopDetector(
        PoseGraphOptimizer& optimizer, 
        std::vector<Frame*>& cam_frames, 
        std::vector<PoseGraphEdge>& edges);

    ~LoopDetector();

    void Track(const std::vector<cv::Mat>& features);
    void Query(const std::vector<cv::Mat>& features);
    void LoadVocabulary(const std::string& file_path);
    void SaveVocabulary(const std::string& file_path) const;

private:
    // p(S = score)
    float ScoreProbability(unsigned int score) const;
    // p(S = score | in_loop)
    float MatchProbability(unsigned int id, float score) const;

private:
    PoseGraphOptimizer&                 m_optimizer;
    std::vector<Frame*>&                m_cam_frames;
    std::vector<PoseGraphEdge>&         m_edges;
    std::vector<std::pair<int, float>>  m_matches;
    std::vector<PoseGraphEdge>          m_potential_edges;

    std::vector<float>                  m_probabilities;

    OrbVocabulary                       m_vocabulary;
    OrbDatabase                         m_database;

    // track the distribution of match score of all adjacent frames
    // to compute match probability for distant frames
    std::vector<float>                  m_scores;
    float                               m_score_avg = 0.0;
    float                               m_score_var = 0.0;

    // p(in_loop | S)
    float                               m_loop_probability = MIN_LOOP_PROBABILITY;

    bool                                m_in_loop = false;
};
