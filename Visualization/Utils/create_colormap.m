function colors = create_colormap(M,N)

% create colormap
minx = min(min(M.VERT(:,1)),min(N.VERT(:,1)));
miny = min(min(M.VERT(:,2)),min(N.VERT(:,2)));
minz = min(min(M.VERT(:,3)),min(N.VERT(:,3)));
maxx = max(max(M.VERT(:,1)),max(N.VERT(:,1)));
maxy = max(max(M.VERT(:,2)),max(N.VERT(:,2)));
maxz = max(max(M.VERT(:,3)),max(N.VERT(:,3)));
colors = [(M.VERT(:,1)-minx)/(maxx-minx) (M.VERT(:,2)-miny)/(maxy-miny) (M.VERT(:,3)-minz)/(maxz-minz)];

end