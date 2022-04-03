#include <assert.h>


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


