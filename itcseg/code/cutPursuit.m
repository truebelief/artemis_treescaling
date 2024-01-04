% reg_strength=1.0;%best
% min_nn=5;%best
% edge_stength=30;%best
function [in_component,components]=cutPursuit(pts,PR_minNN,PR_regStrength,PR_edgeStength,mode,speed,verbose,edge_weight,Euv,node_weight)
    n_nodes=size(pts,1);
    if ~exist('Euv','var')
        Mdl = KDTreeSearcher(pts);
%         [min_idxs,Ds] = knnsearch(Mdl,pts,'K',PR_minNN);
        min_idxs = knnsearch(Mdl,pts,'K',PR_minNN);

        %graph
        Eu = (1:n_nodes)';
        Euv=[];
%         for j=1:PR_minNN-1
%             cEuv=[Eu,min_idxs(:,j+1),Ds(:,j+1)];
%             cEuv(cEuv(:,3)>0.3,:)=[];
%             Euv=[Euv;cEuv(:,1:2)];
%         end
        for j=1:PR_minNN-1
            cEuv=[Eu,min_idxs(:,j+1)];
            Euv=[Euv;cEuv];
        end
        Euv=Euv-1;
    end
    
    
    
    if ~exist('edge_weight','var')
        edge_weight = PR_edgeStength*ones(size(Euv,1),1);
    end
    
    if ~exist('node_weight','var')
        node_weight = ones(n_nodes,1);
    end


%     [solution, in_component, components] = L0_cut_pursuit_segmentation(single(current_cluster_pos'), uint32(Euv(:,1)'), uint32(Euv(:,2)'), single(PR_regStrength)...
%         , single(edge_weight), single(node_weight), 1, 0, 2);

    [solution, in_component, components] = L0_cut_pursuit_segmentation(single(pts'), uint32(Euv(:,1)'), uint32(Euv(:,2)'), single(PR_regStrength), single(edge_weight), single(node_weight), mode, speed, verbose);
end
