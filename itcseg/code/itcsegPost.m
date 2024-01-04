close all;

PR_regStrength=10.0;
PR_minNN=6;
PR_edgeStength=1.0;
PR_mergeIterationStep=1.0;


PR_max_attach_dist=2.0;
PR_min_tree_pts=3;
PR_buffer_K=5;%leverage 2D and 3D distance (big 3D gap, but small 2D distance is not preferred)

mode=1;
speed=0;
verbose=0;

tile_size=800;

fpath='..\logs\app\';
foutpath='..\logs\app\';

fls=dir([fpath,'/*.laz']);

fnames={};
for i=1:size(fls,1)
    fn=fls(i).name;
    [~,fnames{i},~]=fileparts(fn);
end

if exist(foutpath,'dir')<1
    mkdir(foutpath);
end



seg_ids_all_counter=0;
for t=1:numel(fnames)
    fprintf('Processing (%d/%d): %s @%s',t,numel(fnames),fnames{t},datestr(now,'yyyy-mm-dd HH:MM:SS.FFF'));
    fname=strcat(fpath,fnames{t},'.laz');
    [hdr,s]  = las2mat(['-i ',fname]);
    s0=s;
    if isfield(hdr,'attributes')
        find_grd_attr_idx=find(ismember({hdr.attributes.name},'VegCls'));
        find_conf_attr_idx=find(ismember({hdr.attributes.name},'ConfPred'));
        scenePts0=[s.x,s.y,s.z,double(s.attributes(:,find_grd_attr_idx)),double(s.attributes(:,find_conf_attr_idx))];

        abg_ind0=scenePts0(:,end-1)>1;
    else
        return;
    end

    seg_ids=zeros(size(scenePts0,1),1);
    seg_ids_max_counter=0;

    [~,tile_u,tile_v]=unique(floor((scenePts0(:,1:3)-min(scenePts0(:,1:3),[],1))/tile_size+1),'rows');
    [tile_v_sorted,tile_v_idx]=sort(tile_v);
    tile_v_group = mat2cell(tile_v_idx, diff([0;find([(diff(tile_v_sorted) ~= 0);1])]),1);
    n_tiles=numel(tile_u);

    for p=1:n_tiles
        scenePts=scenePts0(tile_v_group{p},:);

        abg_ind=abg_ind0(tile_v_group{p});
        scenePts(:,1:3)=scenePts(:,1:3)-min(scenePts(:,1:3),[],1);


        segs_abg=scenePts(abg_ind,:);

        n_nodes=size(segs_abg,1);
        if n_nodes>100
            segs_conf_ind=segs_abg(:,end)==2;
            segs_seeds=segs_abg(segs_conf_ind,:);
            seg_ids_per_tile=zeros(size(segs_abg,1),1);
        
        
            subPos=segs_seeds(:,1:2);
            n_nodes=size(subPos,1);
            Mdl = KDTreeSearcher(subPos);
            [min_idxs,Ds] = knnsearch(Mdl,subPos,'K',PR_minNN);

            %graph
            Eu = (1:n_nodes)';
            Euv=[];
            for j=1:PR_minNN-1            
                cEuv=[Eu,min_idxs(:,j+1),Ds(:,j+1)];
                Euv=[Euv;cEuv(:,1:2)];
            end
            Euv=Euv-1;
            edgeWeight=ones(size(Euv,1),1);
            n_nodes=size(subPos,1);
            if n_nodes>PR_minNN
                [init_in_component,init_components]=cutPursuit(subPos(:,1:2),PR_minNN,PR_regStrength,PR_edgeStength,mode,speed,verbose,edgeWeight,Euv);
                init_seg_idx=double(init_in_component)+1;
            else
                init_seg_idx=1;
            end

            [init_seg_idx_a,init_seg_idx_b]=histc(init_seg_idx,unique(init_seg_idx));
            init_seg_iso_ind=(init_seg_idx_a(init_seg_idx_b)==1);
            init_seg_idx(init_seg_iso_ind)=0;%remove isolated seeds; sometimes big PR_minNN will contain points with big gaps to satisfy the NN requirement
         
            seg_ids_per_tile(segs_conf_ind)=init_seg_idx;
            segs_conf_ind=(seg_ids_per_tile>0);


            segs_abg1=[segs_abg,(1:size(segs_abg,1))',seg_ids_per_tile];

            %attach remaining by (graph pathing x) region growing
            prev_segs_seed_size=0;
            segs_remain=segs_abg1(~segs_conf_ind,:);
            segs_seeds=segs_abg1(segs_conf_ind,:);
            for itr=1:2
                prev_segs_seed_size=size(segs_seeds,1);    
                Mdl2 = KDTreeSearcher(segs_seeds(:,1:2));
                [min_idxs2,D2s] = knnsearch(Mdl2,segs_remain(:,1:2),'K',PR_buffer_K);
            
                D3s=zeros(size(D2s));
                for k=1:PR_buffer_K
                    D3s(:,k)=sqrt(sum((segs_seeds(min_idxs2(:,k),1:3)-segs_remain(:,1:3)).^2,2));
                end
                [~,min_col_idx]=min(D3s+D2s,[],2);
                min_idxs3=min_idxs2(sub2ind(size(min_idxs2),(1:size(min_idxs2,1))',min_col_idx));
                D3s_min=sqrt(sum((segs_seeds(min_idxs3,1:3)-segs_remain(:,1:3)).^2,2));            
            
                filter_ind=D3s_min<PR_mergeIterationStep;
                segs_abg1(segs_remain(filter_ind,end-1),end)=segs_abg1(segs_seeds(min_idxs3(filter_ind),end-1),end);%assign seed id to remain id
                segs_conf_ind(segs_remain(filter_ind,end-1))=true;        

                segs_remain=segs_abg1(~segs_conf_ind,:);
                segs_seeds=segs_abg1(segs_conf_ind,:);
            end        

            segs_remain=segs_abg1(~segs_conf_ind,:);
            segs_seeds=segs_abg1(segs_conf_ind,:);
            Mdl2 = KDTreeSearcher(segs_seeds(:,1:2));
            [min_idxs2,D2s] = knnsearch(Mdl2,segs_remain(:,1:2),'K',PR_buffer_K);
        
            D3s=zeros(size(D2s));
            for k=1:PR_buffer_K
                D3s(:,k)=sqrt(sum((segs_seeds(min_idxs2(:,k),1:3)-segs_remain(:,1:3)).^2,2));
            end
            [~,min_col_idx]=min(D3s+D2s,[],2);
            min_idxs3=min_idxs2(sub2ind(size(min_idxs2),(1:size(min_idxs2,1))',min_col_idx));
            D3s_min=sqrt(sum((segs_seeds(min_idxs3,1:3)-segs_remain(:,1:3)).^2,2));
            D2s_min=sqrt(sum((segs_seeds(min_idxs3,1:2)-segs_remain(:,1:2)).^2,2));
        
            filter_ind=0.5*(D3s_min+D2s_min)<PR_max_attach_dist;
            segs_abg1(segs_remain(filter_ind,end-1),end)=segs_abg1(segs_seeds(min_idxs3(filter_ind),end-1),end);%assign seed id to remain id


            [seg_u,~,seg_v]=unique(segs_abg1(:,end));
            [seg_v_sorted,seg_v_idx]=sort(seg_v);
            seg_v_group = mat2cell(seg_v_idx, diff([0;find([(diff(seg_v_sorted) ~= 0);1])]),1);
        
            segs_abg1(:,end)=0;
            n_segs=numel(seg_u);
            segs_counter=1;
            for i=2:n_segs
                if numel(seg_v_group{i})>=PR_min_tree_pts
                    segs_abg1(seg_v_group{i},end)=segs_counter;
                    segs_counter=segs_counter+1;
                end
            end

            seg_ids_per_tile=segs_abg1(:,end);
            max_counter=max(seg_ids_per_tile);
    
            ind=seg_ids_per_tile>0;
            seg_ids_per_tile(ind)=seg_ids_per_tile(ind)+seg_ids_max_counter;
            seg_ids(tile_v_group{p}(abg_ind))=seg_ids_per_tile;
            seg_ids_max_counter=seg_ids_max_counter+max_counter;

        end
    end

    pred_file=strcat(foutpath,fnames{t},'_segs.laz');
    s=s0;
    new_hdr(1,1) = struct(...
    'name', 'segs', ...
    'type', 6, ... 
    'description', 'segs', ...
    'scale', 1.0, ...
    'offset', 0 ...
    );
    %type 6=long, type 7=unsigned long long (unsupported), type 8=long long (unsupported), type 9=float, type 10=double

    if isfield(hdr,'attributes')
        ex_hdr=[hdr.attributes,new_hdr];
    else
        ex_hdr=new_hdr;
        s.attributes=[];
    end
    for k=1:size(ex_hdr,2)
        ex_hdr(:,k).scale=1.0;
    end
    s.attribute_info = ex_hdr;
    s.attributes(:,end+1) = seg_ids;
    mat2las(s,[' -o ',pred_file, sprintf(' -fgi_scale %.5f %.5f %.5f',hdr.x_scale_factor,hdr.y_scale_factor,hdr.z_scale_factor),  sprintf(' -fgi_offset %.5f %.5f %.5f',hdr.x_offset,hdr.y_offset,hdr.z_offset)]);

    fprintf(' @%s\n',datestr(now,'yyyy-mm-dd HH:MM:SS.FFF'));
end


% miou_avg=miou_avg/numel(fnames);dr_avg=dr_avg/numel(fnames);
% fprintf('\nFinal mean mIoU: %.3f; rate(%%): %.3f \n',miou_avg,dr_avg);
