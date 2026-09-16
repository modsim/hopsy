#ifndef PROPOSAL_H
#define PROPOSAL_H

template <typename State, typename Extra = std::monostate>
struct Proposal {
    State state;
    [[no_unique_address]] Extra extra;
}

#endif //PROPOSAL_H
