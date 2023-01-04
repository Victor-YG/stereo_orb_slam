#include <assert.h>


// *******************
// * visual odometer *
// *******************

// number of features to detect
const unsigned int NUM_FEATURES = 1000;

// feature appearance based matching distance threshold
const float FEATURE_MATCHING_DISTANCE_THRES = 30.0;

// feature appearance based matching ratio threshold in lowe's test
const float FEATURE_MATCHING_RATIO_THRES = 0.5;

// whether conduct a final model fitting with all inlier at the end
const bool RANSAC_FINAL_MODEL_FITTING = false;

// whether terminal model seach after a concensus reached (inlier more than certain percentage)
const bool RANSAC_EARLY_TERMINATION = false;

// the target concensus percentage (between 0 and 1) for early termination
const float RANSAC_CONCENSUS_RATIO = 0.8;

// maximum search iteration
const unsigned int RANSAC_MAX_ITERATION = 100;

// *********************
// * bundle adjustment *
// *********************

// maximum refinement iteration for bundle adjustment (local and global)
const unsigned int BA_MAX_ITERATION = 50;

// number of threads
const unsigned int BA_NUM_THREADS = 4;

// max_solver_time_in_seconds
// const float BA_MAX_TIME_SEC = 1e32;
const float BA_MAX_TIME_SEC = 1e0;

// parameter lower bound
const float BA_POINT_COORD_LOWER_BOUND = -10000.0;

// parameter upper bound
const float BA_POINT_COORD_UPPER_BOUND = 10000.0;

// ****************
// * loop closure *
// ****************

// how many frame candidates to return when query from bag of words database
const unsigned int QUERY_SIZE_FROM_BOW_DATABASE = 4;

// threshold in frame index difference to determine whether two frames are adjacent
const unsigned int THRES_ADJACENT_FRAME = 5;

// threshold in frame index difference to dtermine whether two frames are distant
const unsigned int THRES_DISTANCE_FRAME = 50;

// how many consecutive frames around target frame to use to determine the match probability
const unsigned int SCORE_WINDOW = 5;

// threshold in probability when a distance frame is a good match
const float THRES_MATCH_PROBABILITY = 0.5;

// threshold in probability when a loop is detected
const float THRES_IS_LOOP = 0.9;

// threshold in probability when a possible loop is denied
const float THRES_NOT_LOOP = 0.1;

// decay rate of loop probability when no match is found
const float DECAY_RATE = 0.75;

// minimum loop probability such that it takes finite # of frames to register a new loop
const float MIN_LOOP_PROBABILITY = 0.005;

// maximum loop probability such that it takes finite # of frames to reject a loop
const float MAX_LOOP_PROBABILITY = 0.995;
