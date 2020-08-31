#!/usr/bin/perl

# Copyright (c) 2012 Albrecht Lindner (AlbrechtJLindner [at] gmail [dot] com)
# All rights reserved
#
# License: Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)
# 
# If you are using (parts of) this code, please cite the corresponding publication:
# Albrecht Lindner, Bryan Zhi Li, Nicolas Bonnier, and Sabine Süsstrunk, A large-scale multi-lingual color thesaurus, IS&T Color and Imaging Conference, 2012.


use strict;
use Switch;

my $num_args = $#ARGV + 1;
if ($num_args != 2) {
  print "\nUsage: google_download.pl directory language\n";
  exit;
}

# set and display arguments
my $DIR = $ARGV[0];
my $LANG = $ARGV[1];
my $XKCD_TEXT = "";
my $COUNTRY = "";
my $EXTENSION = "jpg";
my $COLOR = "";

switch ($LANG) {
	case "lang_en" {$XKCD_TEXT = "./xkcd_text/english.txt"; $COUNTRY = "countryUS"; $COLOR = "color";}
	case "lang_de" {$XKCD_TEXT = "./xkcd_text/german.txt"; $COUNTRY = "countryDE"; $COLOR = "farbe";}
	case "lang_fr" {$XKCD_TEXT = "./xkcd_text/french.txt"; $COUNTRY = "countryFR"; $COLOR = "couleur";}
	case "lang_pt" {$XKCD_TEXT = "./xkcd_text/portuguese.txt"; $COUNTRY = "countryPT"; $COLOR = "cor";}
	case "lang_zh-CN" {$XKCD_TEXT = "./xkcd_text/chinese.txt"; $COUNTRY = "countryCN"; $COLOR = "";}
	case "lang_ko" {$XKCD_TEXT = "./xkcd_text/korean.txt"; $COUNTRY = "countryKR"; $COLOR = "";}
	case "lang_es" {$XKCD_TEXT = "./xkcd_text/spanish.txt"; $COUNTRY = "countryES"; $COLOR = "color";}
	case "lang_it" {$XKCD_TEXT = "./xkcd_text/italian.txt"; $COUNTRY = "countryIT"; $COLOR = "colore";}
	case "lang_ru" {$XKCD_TEXT = "./xkcd_text/russian.txt"; $COUNTRY = "countryRU"; $COLOR = "";}
	case "lang_jp" {$XKCD_TEXT = "./xkcd_text/japanese.txt"; $COUNTRY = "countryJP"; $COLOR = "";}
}

print "XKCD_TEXT = $XKCD_TEXT\n";
print "LANG = $LANG\n";
print "COUNTRY = $COUNTRY\n";
print "DIR = $DIR\n";
print "EXTENSION = $EXTENSION\n";

# prepare directories
my $TMPDIR = "./tmp_$LANG";
print "TMPDIR = $TMPDIR\n";
unless(-d $TMPDIR){
    mkdir $TMPDIR or die;
}
my $QUERYDIR = "$DIR$LANG/queries/";
unless(-d "$DIR$LANG"){
    mkdir "$DIR$LANG" or die "cannot create ".$DIR.$LANG."\n";
}
unless(-d $QUERYDIR){
    mkdir $QUERYDIR or die "cannot create ".$QUERYDIR."\n";
}
my $COLORDIR = "$DIR$LANG/colors/";
unless(-d $COLORDIR){
    mkdir $COLORDIR or die "cannot create ".$COLORDIR."\n";
}


# read xkcd text files
open FILE, "<", "$XKCD_TEXT" or die $!;
my @XKCD = <FILE>;
close FILE;
open FILE, "<", "./xkcd_text/english.txt" or die $!;
my @XKCD_EN = <FILE>;
close FILE;

my $N = @XKCD;
print "$N lines in file $XKCD_TEXT\n";

my $cmd = "";
my $i;
for ($i = 0; $i < $N; $i++) {
	my $cname = $XKCD[$i];
	$cname =~ s/\n|\r//g;
	my $cname_EN = $XKCD_EN[$i];
	$cname_EN =~ s/\n|\r//g;
	print "$i: \"$cname\" \"$cname_EN\"\n";
	
	$cmd = "rm $TMPDIR/*";
	print "-> $cmd\n";
	`$cmd`;
	
	my @all_files = qw();
	# download for current color
	for (my $j = 0; $j < 5; $j++) {
		my $OFFSET = 20*$j;
		if ($COLOR =~ m/.+/) {
			$cmd = "wget -U firefox -O $TMPDIR/orig.txt http://images.google.com/images?q=\"$cname\"+$COLOR\\&lr=$LANG\\&cr=$COUNTRY\\&as_filetype=$EXTENSION\\&start=$OFFSET";
		}
		else {
			$cmd = "wget -U firefox -O $TMPDIR/orig.txt http://images.google.com/images?q=\"$cname\"\\&lr=$LANG\\&cr=$COUNTRY\\&as_filetype=$EXTENSION\\&start=$OFFSET";
		}
		$cmd = "$cmd > \"$QUERYDIR$cname_EN$OFFSET.txt\"";
		print "-> $cmd\n";
		`$cmd`;
		$cmd = "cat $TMPDIR/orig.txt | grep imgurl > $TMPDIR/step1.txt";
		print "-> $cmd\n";
		`$cmd`;
		$cmd = "cat $TMPDIR/step1.txt | tr \"&\" \"\n\" > $TMPDIR/step2.txt";
		print "-> $cmd\n";
		`$cmd`;
		$cmd = "cat $TMPDIR/step2.txt | grep \"imgres?imgurl\" > $TMPDIR/step3.txt";
		print "-> $cmd\n";
		`$cmd`;
		$cmd = "cat $TMPDIR/step3.txt | tr \"><\" \"\\n\" > $TMPDIR/step4.txt";
		print "-> $cmd\n";
		`$cmd`;
		$cmd = "cat $TMPDIR/step4.txt | grep \"imgres?imgurl\" > $TMPDIR/step5.txt";
		print "-> $cmd\n";
		`$cmd`;
		$cmd = "cat $TMPDIR/step5.txt | tr \"=\" \"\\n\" > $TMPDIR/step6.txt";
		print "-> $cmd\n";
		`$cmd`;
		$cmd = "cat $TMPDIR/step6.txt | grep \"http\" > $TMPDIR/step7.txt";
		print "-> $cmd\n";
		`$cmd`;
		
		$cmd = "wget --directory-prefix=$TMPDIR/ --tries=1 --timeout=60 -i $TMPDIR/step7.txt -U firefox";
		print "-> $cmd\n";
		`$cmd`;
		
# this is a pause		
# my $age = <>;
		
		open FILES, "<", "$TMPDIR/step7.txt" or die $!;
		my @files = <FILES>;
		close FILES;
		push(@all_files, @files);
		
		$cmd = "mv $TMPDIR/orig.txt \"$QUERYDIR$cname_EN$OFFSET.html\"";
		print "-> $cmd\n";
		`$cmd`;
		$cmd = "mv $TMPDIR/step7.txt \"$QUERYDIR$cname_EN$OFFSET.txt\"";
		print "-> $cmd\n";
		`$cmd`;
#		my $age = <>;
	}
#	print "press something to continue\n";
#	my $age = <>;
	
#	for i in 0 1 2
#do
#	OFFSET=`echo ${i}*20 | bc`
#	wget -U firefox -O orig.txt http://images.google.com/images?q=${SEARCH}+color\&as_filetype=${EXTENSION}\&start=${OFFSET} > "../../queries/${DIR}_${OFFSET}.txt"
#		cat orig.txt | grep imgurl > step1.txt
#		cat step1.txt | tr "&" "\n" > step2.txt
#		cat step2.txt | grep .gstatic.com > step3.txt
#		cat step3.txt | tr "><" "\n" > step4.txt
#		cat step4.txt | tr "\"" "\n" > step5.txt
#		cat step5.txt | grep http > step6.txt
#	wget --tries=1 --timeout=60 -i step6.txt -U firefox
#	mv orig.txt "../../queries/${DIR}_${OFFSET}.html"
#done
	
#	sleep(60);
	
	
	print "FINISHING $cname_EN=============================================\n";
	# delete intermediate files
	$cmd = "rm $TMPDIR/step?.txt";
	print "-> $cmd\n";
	`$cmd`;
	# mv all files to respective directory
	unless(-d "$COLORDIR$cname_EN"){
	   mkdir "$COLORDIR$cname_EN" or die;
	}
	
#	print "@all_files\n";
	
# copy all files from tmp into respective color folder
	my $N = @all_files;
	my $p = 1;
	my $counter = 1;
	for (my $p = 1; $p < $N+1; $p++) {
		my $f = $all_files[$p-1];
		$f =~ s/.*\/(.*\.jpg)/$1/;
		$f =~ s/\n|\r//g;
		$cmd = "cp \"$TMPDIR/$f\" \"$COLORDIR$cname_EN/$counter.jpg\"";
		print "-> $cmd\n";
		system($cmd);
		if ($? == 0) { # copy command was successful
			$counter = $counter + 1;
			print "SUCCESS\n";
		}
		else {
			print "FAILED\n";
		}
	}
#	sleep(3600);
}
