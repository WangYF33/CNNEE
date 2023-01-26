#!/bin/sh
#module load anaconda3/2020.07
#source activate wyf

#parse input parameter
func(){
	echo "Usage:"
	echo "sh get_eRNA.sh [-e enh_file] [-c chrom_sizes] [-f fafile_path] [-g gtffile_path] [-t tissue] [-d dataPath] [-o outputPath] [options]"
	echo "mandatory:"
	echo "  -e  FILE      enter enh_file,the enhancer location file with path,such as: path/enhacner.txt"
	echo "  -c  FILE      enter chrom_sizes,the chrom sizes file with path,which can be downloaded from UCSC, such as: path/xx.chrom.sizes "
	echo "  -f  FILE      enter faFile, the reference fa file with the path,such as: path/XXX.fa"
	echo "  -g  FILE      enter gtffile_path, the reference gtf file with the path,such as: path/xxx.gtf"
	echo "  -t  STR       if have tissue name,enter -I tissue ; if no have tissue name , enter -I 0"
	echo "  -d  PATH	enter dataPath, the path of data, such as ./path"
	echo "  -o  PATH	enter outputPath, the path to the output, such as ./path"
	echo "  -p  FILE      enter pcg_file,the protein coding gene file with path,such as: path/pcg.txt"
	echo "options:"
	echo "  -s  PATH	optional parameter,enter the path of the eRNA_RPM.R, such as ./path"
	echo "  -a  number	optional parameter, set the thread. Default 24"
	exit -1
}

index=1
SRR=1
bam=1
thread=24
code_path=0
  	
while getopts "e:c:f:g:t:d:o:s:p:h:IRBh" opt; do
  case $opt in
    e) enh_file=$OPTARG   ;; #enhancer_file=`cat ${enh_file}`
    c) chrom_sizes=$OPTARG    ;; #chrom_sizes="susScr11.chrom.sizes.txt"
  	f) fafile_path=$OPTARG   ;;   
  	g) gtffile_path=$OPTARG   ;;   
  	t) tissue=$OPTARG   ;;  #tissue name
  	d) dataPath=$OPTARG   ;;   # File name and path of the intput data  
  	o) outputPath=$OPTARG   ;;    # File name and path of the output result  
	p) pcg_file=$OPTARG   ;;  ##
	s) code_path=$OPTARG   ;;
	a) thread=$OPTARG   ;; 
	  h) func	;;	#  help
	  ?) func	;;	#error command,print help
  esac
done
####### file_path
module load R
module load hisat2
module load samtools  
module load sra-toolkit/2.8.2-1-gcc-4.8.5
fastp="/BIGDATA2/scau_xlyuan_1/software/fastp"


index_name=$(basename $fafile_path .fa)
index_path=${fafile_path%/*}

outputPath="${outputPath}"
dataPath="${dataPath}"
mkdir -p ${outputPath}/temp
tempPath="${outputPath}/temp"
mkdir -p ${dataPath}/cleandata
cleandata="${dataPath}/cleandata"
mkdir -p ${dataPath}/temp
datatempPath="${dataPath}/temp"



cd ${tempPath}
raw_region="_raw_eRNA_region.txt"
touch "$tissue$raw_region"
raw_eRNA_region="$tissue$raw_region"

num=`cat ${enh_file} | wc -l`
#num=$(($num+1))
for((j=1;j<=$num;j++))
do
  #通过awk取出第j行第二列的内容
  chr=$(awk -v co="${j}" 'NR==co {print $1}' ${enh_file})
  start=$(awk -v co="${j}" 'NR==co {print $2}' ${enh_file})
  end=$(awk -v co="${j}" 'NR==co {print $3}'  ${enh_file})
  enh_median1=$((($start+$end)/2))
  enh_median2=$(((($start+$end)/2)+1))
  #将计算结果保存到end_result中
  echo -e "${chr}\t${enh_median1}\t${enh_median2}" > ${tempPath}/${tissue}_end_result
  bedtools slop -i ${tissue}_end_result -g ${chrom_sizes} -l 3000 -r 3000 >>  ${tempPath}/raw_eRNA.txt
  bedtools subtract -a   ${tempPath}/raw_eRNA.txt -b ${pcg_file} -A > ${outputPath}/${raw_eRNA_region}
done
##
cd ${bamPath}
ls *.bam | awk '{split($1, arr, "."); print arr[1]}' > ${tempPath}/samplename.txt
bamnametmp="${tempPath}/samplename.txt"
bamname=($(awk '{print $1}' ${bamnametmp}))
for((i=0;i<${#bamname[@]};i++))
do
cd ${index_path}
  extract_splice_sites.py ${gtffile_path} > ${index_path}/${index_name}.ss
  extract_exons.py ${gtffile_path} > ${index_path}/${index_name}.exon
  hisat2-build -p ${thread} --ss ${index_path}/${index_name}.ss --exon ${index_path}/${index_name}.exon ${faFile} ${index_path}/${index_name}
  hisat2-build -f ${faFile} ${index_path}/${index_name}
  # add chrom to bam
  samtools view -H ${bamPath}/${bamname[i]}.bam | sed -e 's/SN:\([0-9XY]\)/SN:chr\1/' -e 's/SN:MT/SN:chrM/' | samtools reheader - ${bamPath}/${bamname[i]}.bam > ${tempPath}/chr-${bamname[i]}.bam
  samtools view -bq 1 ${tempPath}/chr-${bamname[i]}.bam > ${tempPath}/unique-${bamname[i]}.bam
  samtools index ${tempPath}/unique-${bamname[i]}.bam
  bedtools multicov -bams ${tempPath}/unique-${bamname[i]}.bam -bed ${outputPath}/${raw_eRNA_region} > ${tempPath}/${bamname[i]}-map.txt
  samtools view -c -f 1 -F 12 ${tempPath}/chr-${bamname[i]}.bam > ${tempPath}/total.count_${bamname[i]}.txt
done
ls ${tempPath}/*bam.bai | awk 'BEGIN{ORS="\n"} {print $0}' > ${tempPath}/index_name.tmp
indexnametmp="${tempPath}/index_name.tmp"
index_name=($(awk '{print $1}' ${indexnametmp}))
if (( ${#index_name[@]} == ${#bamname[@]} ))
   then
   	rm *bam.bai
   else
    echo "index number error"
fi

cd ${tempPath}
ls *-map.txt | awk 'BEGIN{ORS="\n"} {print $0}' > ${tempPath}/eRNA_map_count.txt
erna_map_tmp="${tempPath}/eRNA_map_count.txt"
erna_map=($(awk '{print $1}' ${erna_map_tmp}))
touch out.txt
out="out.txt"
for(( i=0;i<${#erna_map[@]};i++))
do
  f=$i.tmp
  j=$i.tmpp
  awk '{print $4}' ${erna_map[i]} > erna$f.txt
  erna=erna$f.txt
  paste ${out} ${erna} > erna_merge$j.txt
  out=erna_merge$j.txt
  if (($i > 0))
  	then
  		rm erna_merge$(((i)-1)).tmpp.txt
  fi
  rm erna$i.tmp.txt
done
ls *map.txt | awk 'BEGIN{ORS="\n"} {print $0}' > ${tempPath}/map_name.tmp
mapnametmp="${tempPath}/map_name.tmp"
map_name=($(awk '{print $1}' ${mapnametmp}))
if (( ${#erna_map[@]} == ${#bamname[@]} ))
   then
   	rm *map.txt
   else
    echo "map_file number error"
fi

ls total.count_* | awk 'BEGIN{ORS="\n"} {print $0}' > ${tempPath}/total.count.tmp
a_path="${tempPath}/total.count.tmp"
total_count=($(awk '{print $1}' ${a_path}))
touch "total-out.txt"
out="total-out.txt"
for((i=0;i<${#total_count[@]};i++))
do
    j=$i.tmp
    paste ${out} ${total_count[i]}  > totalcount$j.txt
    out=totalcount$j.txt
    if (($i > 0))
    	then    
        rm totalcount$(((i)-1)).tmp.txt  
      else
        echo "totalcount error"
    fi      
done

###row to col
cat ${tempPath}/samplename.txt | awk 'BEGIN{c=0;} {for(i=1;i<=NF;i++) {num[c,i] = $i;} c++;} END{ for(i=1;i<=NF;i++){str=""; for(j=0;j<NR;j++){ if(j>0){str = str" "} str= str"\t"num[j,i]}printf("%s\n", str)} }' > ${tempPath}/sample_file_name.txt

####merger three col to one col,get eRNA id 
awk '{print $1}' ${outputPath}/${raw_eRNA_region} > ${tempPath}/erna_chr.tmp
awk '{print $2}' ${outputPath}/${raw_eRNA_region} > ${tempPath}/erna_start.tmp
awk '{print $3}' ${outputPath}/${raw_eRNA_region} > ${tempPath}/erna_end.tmp
paste -d ':-' ${tempPath}/erna_chr.tmp ${tempPath}/erna_start.tmp ${tempPath}/erna_end.tmp > ${outputPath}/${tissue}_raw_eRNA_id.txt

sed -i '1i\bam_total_count' ${outputPath}/${tissue}_raw_eRNA_id.txt
sed -i '1i\name' ${outputPath}/${tissue}_raw_eRNA_id.txt
map_path="${tempPath}/eRNA_map_count.txt"
map_num=($(awk '{print $1}' ${map_path}))
cat sample_file_name.txt totalcount$((${#map_num[@]}-1)).tmp.txt erna_merge$((${#map_num[@]}-1)).tmpp.txt >> ${tempPath}/${tissue}_map_count_tmp.txt
paste ${outputPath}/${tissue}_raw_eRNA_id.txt ${tempPath}/${tissue}_map_count_tmp.txt | sed 's/\t\t/\t/g' > ${tempPath}/${tissue}_eRNA_bam_mapcounts.tmp
awk '{ 
    for (i=1; i<=NF; i++){
        if(NR==1){ 
            # When processing the first row, the value of column i ($i) is stored in arr[i], i is the subscript of the array, and the array can be used directly without definition
            arr[i]=$i;   
        }
        else{
            # When it is not the first row, stitch the values of the row corresponding to the i column to arr[i]
            arr[i]=arr[i] " " $i
        }
    }
}
END{
    # 每行处理完以后,输出数组
    for (i=1; i<=NF; i++){
        print arr[i]
    }
}' ${tempPath}/${tissue}_eRNA_bam_mapcounts.tmp > ${tempPath}/${tissue}.eRNA_mapcount_row2col.txt
sed -i 's/[ ]/\t/g' ${tempPath}/${tissue}.eRNA_mapcount_row2col.txt
cd ${outputPath}
if (( $code_path == 0 ))
	then
    code_path=${outputPath}
fi
rpm_input="${tempPath}/${tissue}.eRNA_mapcount_row2col.txt"
tmp_file="${tempPath}/${tissue}_rpm_tmp.txt"
rpm_output="${tempPath}/${tissue}_enhancer_RPM.txt"
eRNA_RPM="${outputPath}/${tissue}_eRNA_rpm_mean.txt"
eRNA_file="${outputPath}/${tissue}_eRNA_region.txt"
Rscript ${code_path}/eRNA_RPM.R ${rpm_input} ${tmp_file} ${rpm_output} ${eRNA_RPM} ${eRNA_file}


