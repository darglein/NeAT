/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#include "ortools/linear_solver/linear_solver.h"
//
#include "HyperTree.h"
#include "HyperTreeStructureOptimizer.h"
HyperTreeStructureOptimizer::HyperTreeStructureOptimizer(HyperTreeBase tree, TreeOptimizerParams params)
    : tree(tree), params(params)
{
    solver = std::make_shared<operations_research::MPSolver>(
        "CBC", operations_research::MPSolver::OptimizationProblemType::CBC_MIXED_INTEGER_PROGRAMMING);

    auto res = solver->SetNumThreads(params.num_threads);
    if (!res.ok())
    {
        std::cout << "solver->SetNumThreads failed with message " << res.message() << std::endl;
    }

    data.resize(tree->NumNodes());

    CreateVariables();
    CreateConstraints();
    CreateObjective();
}
bool HyperTreeStructureOptimizer::OptimizeTree()
{
    std::cout << "Optimizing Tree Structure" << std::endl;

    std::cout << "> #Constraints " << solver->NumConstraints() << " #Variables " << solver->NumVariables()
              << " #Threads " << solver->GetNumThreads() << std::endl;

    operations_research::MPSolver::ResultStatus result_status = solver->Solve();

    std::cout << "> Result Status: " << result_status << std::endl;
    std::cout << "> Objective: " << solver->Objective().Value() << std::endl;

    int num_none  = 0;
    int num_grp   = 0;
    int num_split = 0;
    int num_nodes = 0;

    for (PerNodeData& nd : data)
    {
        if (nd.I_none)
        {
            num_none += round(nd.I_none->solution_value());
            num_grp += round(nd.I_grp->solution_value());
            num_split += round(nd.I_split->solution_value());
            num_nodes++;
        }
    }

    std::cout << "> N | None/Merge/Split: " << num_nodes << " | " << num_none << "/" << num_grp << "/" << num_split
              << std::endl;


    ApplySplit();

    return num_none != num_nodes;
}
void HyperTreeStructureOptimizer::CreateVariables()
{
    CHECK(tree->active_node_ids.is_cpu());
    for (int i = 0; i < tree->NumActiveNodes(); ++i)
    {
        int node_id            = tree->active_node_ids.data_ptr<long>()[i];
        PerNodeData& node_data = data[node_id];

        int parent_id     = tree->node_parent.data_ptr<int>()[node_id];
        int* children_ids = tree->node_children.data_ptr<int>() + node_id * tree->node_children.stride(0);


        CHECK(node_data.I_none == nullptr);
        node_data.I_split = solver->MakeBoolVar("");
        node_data.I_none  = solver->MakeBoolVar("");
        node_data.I_grp   = solver->MakeBoolVar("");
        CHECK_NOTNULL(node_data.I_none);


        if (parent_id < 0)
        {
            // The root is not allowed to merge
            node_data.I_grp->SetBounds(0, 0);
        }

        if (children_ids[0] < 0)
        {
            // Leaf nodes are not allowed to split
            node_data.I_split->SetBounds(0, 0);
        }
    }
}
void HyperTreeStructureOptimizer::CreateConstraints()
{
    using namespace operations_research;

    auto culled       = tree->node_culled.cpu();
    int* culling_flag = culled.data_ptr<int>();
    int* active_flag  = tree->node_active.data_ptr<int>();

    int NS                    = tree->NS();
    double fac                = 1;
    auto max_patch_constraint = solver->MakeRowConstraint(-solver->infinity(), params.max_active_nodes * fac);
    for (int i = 0; i < tree->NumActiveNodes(); ++i)
    {
        int node_id            = tree->active_node_ids.data_ptr<long>()[i];
        PerNodeData& node_data = data[node_id];

        int parent_id     = tree->node_parent.data_ptr<int>()[node_id];
        int* sibling_ids  = tree->node_children.data_ptr<int>() + parent_id * tree->node_children.stride(0);
        int* children_ids = tree->node_children.data_ptr<int>() + node_id * tree->node_children.stride(0);
        CHECK_NOTNULL(node_data.I_split);

        // 1. The unique constraint
        //    Constraint that guarantees exactly one of the 3 modifiers is set.
        solver->MakeRowConstraint(LinearExpr(node_data.I_split) + LinearExpr(node_data.I_none) +
                                      LinearExpr(node_data.I_grp) ==
                                  LinearExpr(1));


        auto c_merge_split = solver->MakeRowConstraint(fac, fac);
        c_merge_split->SetCoefficient(node_data.I_none, fac);
        max_patch_constraint->SetCoefficient(node_data.I_none, 1 * fac);

        // 2. The merge constraint
        //      All siblings must also set the group variable to 1
        if (parent_id >= 0)
        {
            int num_active_siblings = 0;
            int num_culled_siblings = 0;
            for (int si = 0; si < tree->NS(); ++si)
            {
                int s = sibling_ids[si];
                CHECK_GE(s, 0);
                num_active_siblings += active_flag[s];
                num_culled_siblings += culling_flag[s];
            }

            // Only if this condition is true a merge is possible.
            // If not it means that one of the siblings is further subdivided
            if (num_active_siblings + num_culled_siblings == NS)
            {
                for (int si = 0; si < NS; ++si)
                {
                    int s = sibling_ids[si];
                    CHECK_GE(s, 0);
                    if (!culling_flag[s])
                    {
                        CHECK_NOTNULL(data[s].I_grp);
                        c_merge_split->SetCoefficient(data[s].I_grp, 1.0 / num_active_siblings * fac);
                    }
                }
                max_patch_constraint->SetCoefficient(node_data.I_grp, 1.0 / num_active_siblings * fac);
            }
        }

        // 3. The split constraint
        if (children_ids[0] >= 0)
        {
            int num_culled_children = 0;
            for (int si = 0; si < NS; ++si)
            {
                int c = children_ids[si];
                CHECK_GE(c, 0);
                num_culled_children += culling_flag[c];
            }

            CHECK_LT(num_culled_children, NS);
            c_merge_split->SetCoefficient(node_data.I_split, fac);
            max_patch_constraint->SetCoefficient(node_data.I_split, (NS - num_culled_children) * fac);
        }
    }
}
void HyperTreeStructureOptimizer::CreateObjective()
{
    bool use_old_errors = params.use_saved_errors;

    operations_research::MPObjective* objective = solver->MutableObjective();
    objective->minimization();


    //    auto error_copy   = (tree->node_error * tree->node_max_density);
    //    float* all_errors = error_copy.data_ptr<float>();
    float* all_density = tree->node_max_density.data_ptr<float>();
    float* all_errors  = tree->node_error.data_ptr<float>();

    auto culled       = tree->node_culled.cpu();
    int* culling_flag = culled.data_ptr<int>();
    int* active_flag  = tree->node_active.data_ptr<int>();
    // auto culled       = tree->node_culled.cpu();
    // int* culling_flag = culled.data_ptr<int>();

    for (int i = 0; i < tree->NumActiveNodes(); ++i)
    {
        int node_id            = tree->active_node_ids.data_ptr<long>()[i];
        PerNodeData& node_data = data[node_id];

        int parent_id     = tree->node_parent.data_ptr<int>()[node_id];
        int* sibling_ids  = tree->node_children.data_ptr<int>() + parent_id * tree->node_children.stride(0);
        int* children_ids = tree->node_children.data_ptr<int>() + node_id * tree->node_children.stride(0);


        float max_density_none  = all_density[node_id];
        float max_density_split = max_density_none;
        float max_density_merge = 0;  // computed from siblings

        float loss_none = all_errors[node_id];
        float loss_grp;    // computed from siblings
        float loss_split;  // computed from children

        CHECK(std::isfinite(loss_none));
        CHECK_GE(loss_none, 0);
        CHECK_GE(max_density_none, 0);
        CHECK_GT(tree->node_depth.data_ptr<int>()[node_id], 0);

        {
            // This node is the first child of the parent
            // if (node_id == sibling_ids[0])
            {
                float sibling_error_sum  = 0;
                int num_non_culled_nodes = 0;
                for (int ci = 0; ci < tree->NS(); ++ci)
                {
                    int s = sibling_ids[ci];
                    if (!culling_flag[s] && active_flag[s])
                    {
                        CHECK(active_flag[s]) << "node is not culled and not active. this is not possible.";
                        auto sib_err  = all_errors[s];
                        auto sib_dens = all_density[s];

                        // can happen if a node is not seen but culling is disabled.
                        if (sib_err < 0) sib_err = 0;

                        CHECK_GE(sib_err, 0);
                        sibling_error_sum += sib_err;
                        max_density_merge = std::max(sib_dens, max_density_merge);
                        num_non_culled_nodes++;
                    }
                }
                // instead of the actual error sum we take the maximum for all nodes
                // sibling_error_sum = sibling_error_max * num_non_culled_nodes;

                double predicted_merge_error = params.error_merge_factor * sibling_error_sum;

                {
                    // Prediction if no previous result is available.
                    // The error slightly increases because we have less resolution
                    loss_grp = predicted_merge_error;
                }

                // every sibling computes this error so divide here by the count
                loss_grp /= num_non_culled_nodes;
            }
        }



        // if there is another layer below
        if (children_ids[0] >= 0)
        {
            if (use_old_errors && all_errors[children_ids[0]] > 0)
            {
                max_density_split = 1;
                loss_split = 0;
                for (int ci = 0; ci < tree->NS(); ++ci)
                {
                    int c               = children_ids[ci];
                    float child_err     = all_errors[c];
                    float child_density = all_density[c];
                    // child_err      = std::max(child_err, 0.f);
                    CHECK_GE(child_err, 0);
                    CHECK_GE(child_density, 0);
                    loss_split += child_err * child_density;
                }

                // this is a unrealistic szenario, because a split should always have less error then no split
                // -> just pretend here that the stored error is invalid
                // if (loss_split > loss_none)
                // {
                //     loss_split = params.error_split_factor * loss_none;
                // }
            }
            else
            {
                // Prediction is that the loss slightly decreases after splitting
                loss_split = params.error_split_factor * loss_none;
            }
        }

        operations_research::LinearExpr total(0.0);
        total += operations_research::LinearExpr(node_data.I_none) * loss_none * max_density_none;
        total += operations_research::LinearExpr(node_data.I_split) * loss_split * max_density_split;
        total += operations_research::LinearExpr(node_data.I_grp) * loss_grp * max_density_merge;
        objective->AddLinearExpr(total);
    }
}
void HyperTreeStructureOptimizer::ApplySplit()
{
    // CHECK(tree->node_culled.is_cpu());
    auto culled       = tree->node_culled.cpu();
    int* active_flag  = tree->node_active.data_ptr<int>();
    int* culling_flag = culled.data_ptr<int>();

    for (int i = 0; i < tree->NumActiveNodes(); ++i)
    {
        int node_id            = tree->active_node_ids.data_ptr<long>()[i];
        PerNodeData& node_data = data[node_id];


        int parent_id     = tree->node_parent.data_ptr<int>()[node_id];
        int* children_ids = tree->node_children.data_ptr<int>() + node_id * tree->node_children.stride(0);



        int split = round(node_data.I_split->solution_value());
        int grp   = round(node_data.I_grp->solution_value());
        int none  = round(node_data.I_none->solution_value());

        CHECK_EQ(split + grp + none, 1) << split << " " << grp << " " << none;

        if (split)
        {
            // set children active
            CHECK_GE(children_ids[0], 0);

            active_flag[node_id] = 0;

            for (int ci = 0; ci < tree->NS(); ++ci)
            {
                int c = children_ids[ci];
                if (culling_flag[c] == 0)
                {
                    // only set nodes active that are not culled
                    active_flag[c] = true;
                }
            }
        }
        else if (grp)
        {
            CHECK_GE(parent_id, 0);
            active_flag[node_id]   = 0;
            active_flag[parent_id] = 1;

            // This can not happen because when the parent is culled all its children should also be culled
            CHECK_EQ(culling_flag[parent_id], 0);
        }
        else if (none)
        {
        }
        else
        {
            std::cout << std::endl;
            CHECK(false) << "No split modifier is set!";
        }
    }
    tree->UpdateActive();
}
